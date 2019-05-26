
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import *
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope

def _luong_score(query, keys, scale=True):
  """Implements Luong-style (multiplicative) scoring function.
  This attention has two forms.  The first is standard Luong attention,
  as described in:
  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025
  The second is the scaled form inspired partly by the normalized form of
  Bahdanau attention.
  To enable the second form, call this function with `scale=True`.
  Args:
    query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    scale: Whether to apply a scale to the score function.
  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.
  Raises:
    ValueError: If `key` and `query` depths do not match.
  """
  depth = query.get_shape()[-1]
  key_units = keys.get_shape()[-1]
  if depth != key_units:
    raise ValueError(
        "Incompatible or unknown inner dimensions between query and keys.  "
        "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
        "Perhaps you need to set num_units to the keys' dimension (%s)?"
        % (query, depth, keys, key_units, key_units))
  dtype = query.dtype

  # Reshape from [batch_size, depth] to [batch_size, 1, depth]
  # for matmul.
  query = array_ops.expand_dims(query, 1)

  # Inner product along the query units dimension.
  # matmul shapes: query is [batch_size, 1, depth] and
  #                keys is [batch_size, max_time, depth].
  # the inner product is asked to **transpose keys' inner shape** to get a
  # batched matmul on:
  #   [batch_size, 1, depth] . [batch_size, depth, max_time]
  # resulting in an output shape of:
  #   [batch_time, 1, max_time].
  # we then squeeze out the center singleton dimension.
  score = math_ops.matmul(query, keys, transpose_b=True)
  score = array_ops.squeeze(score, [1])

  if scale:
    # Scalar used in weight scaling
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype, initializer=1.)
    score = g * score
  return score

def _bahdanau_score(processed_query, keys, normalize=True):
  """Implements Bahdanau-style (additive) scoring function.
  This attention has two forms.  The first is Bhandanau attention,
  as described in:
  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473
  The second is the normalized form.  This form is inspired by the
  weight normalization article:
  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868
  To enable the second form, set `normalize=True`.
  Args:
    processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    normalize: Whether to normalize the score function.
  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.
  """
  dtype = processed_query.dtype
  # Get the number of hidden units from the trailing dimension of keys
  num_units = keys.shape[2].value or array_ops.shape(keys)[2]
  # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
  processed_query = array_ops.expand_dims(processed_query, 1)
  v = variable_scope.get_variable(
      "attention_v", [num_units], dtype=dtype)
  if normalize:
    # Scalar used in weight normalization
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype,
        initializer=math.sqrt((1. / num_units)))
    # Bias added prior to the nonlinearity
    b = variable_scope.get_variable(
        "attention_b", [num_units], dtype=dtype,
        initializer=init_ops.zeros_initializer())
    # normed_v = g * v / ||v||
    normed_v = g * v * math_ops.rsqrt(
        math_ops.reduce_sum(math_ops.square(v)))
    return math_ops.reduce_sum(
        normed_v * math_ops.tanh(keys + processed_query + b), [2])
  else:
    return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query), [2])


class SelfAttWrapper(RNNCell):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, cell, initial_attention, initial_memory, att_layer, att_type='B'):
    """Constructs a `ResidualWrapper` for `cell`.
    Args:
      cell: An instance of `RNNCell`.
      residual_fn: (Optional) The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the residual network.
        Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs
        and outputs.
    """
    self._cell = cell
    self._memory_list = [initial_memory,]
    self._attention_list = [initial_attention,]
    assert(att_type=='B' or att_type=='L')
    if att_type == 'B':
      self._att_func = _bahdanau_score
    else:
      self._att_func = _luong_score
    self._att_layer = att_layer

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    inputs = array_ops.concat([inputs, self._attention_list[-1]], 1)
    cell_outputs, new_state = self._cell(inputs, state)
    if self._att_layer is not None:
      query = self._att_layer(cell_outputs)
    else:
      query = cell_outputs
    memory = self._memory_list[-1]
    expand_query = array_ops.expand_dims(query, 1)
    memory = array_ops.concat([memory, expand_query], 1)   # [batch, time+1, depth]
    # [batch_size, 1, depth] * [batch_size, depth, time+1] 
    #expanded_alignments = math_ops.matmul(expand_query, memory, transpose_b=True)   #[batch_size, 1, time+1]
    alignments = self._att_func(query, memory)
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    expanded_attention = math_ops.matmul(expanded_alignments, memory)
    attention = array_ops.squeeze(expanded_attention, [1]) # [batch_size, depth]
    self._attention_list.append(attention)
    self._memory_list.append(memory)
    
    return (cell_outputs, new_state)

class SelfAttOtWrapper(RNNCell):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, cell, initial_memory, att_layer, out_layer, att_type='B'):
    """Constructs a `ResidualWrapper` for `cell`.
    Args:
      cell: An instance of `RNNCell`.
      residual_fn: (Optional) The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the residual network.
        Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs
        and outputs.
    """
    self._cell = cell
    self._memory_list = [initial_memory,]
    self._out_layer = out_layer
    #self._attention_list = [initial_attention,]
    assert(att_type=='B' or att_type=='L')
    if att_type == 'B':
      self._att_func = _bahdanau_score
    else:
      self._att_func = _luong_score
    self._att_layer = att_layer

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    #inputs = array_ops.concat([inputs, self._attention_list[-1]], 1)
    cell_outputs, new_state = self._cell(inputs, state)
    if self._att_layer is not None:
      #print('cell_outputs', cell_outputs)
      query = self._att_layer(cell_outputs)
      #print('query', query)
    else:
      query = cell_outputs
    memory = self._memory_list[-1]
    expand_query = array_ops.expand_dims(query, 1)
    # [batch_size, 1, depth] * [batch_size, depth, time+1] 
    #expanded_alignments = math_ops.matmul(expand_query, memory, transpose_b=True)   #[batch_size, 1, time+1]
    alignments = self._att_func(query, memory)
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    expanded_attention = math_ops.matmul(expanded_alignments, memory)
    attention = array_ops.squeeze(expanded_attention, [1]) # [batch_size, depth]
    #print('attention', attention, 'concat', array_ops.concat([cell_outputs, attention], 1))
    if self._out_layer is not None:
       new_outputs = self._out_layer(array_ops.concat([cell_outputs, attention], 1))
       #print('new_outputs', new_outputs)
    else:
       new_outputs = cell_outputs
    #self._attention_list.append(attention)
    new_memory = array_ops.concat([memory, expand_query], 1)   # [batch, time+1, depth]
    self._memory_list.append(new_memory)
    
    return (new_outputs, new_state)


class SelfAttMulOtWrapper(RNNCell):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, cell, initial_memory, att_layer, out_layer, att_type='B'):
    """Constructs a `ResidualWrapper` for `cell`.
    Args:
      cell: An instance of `RNNCell`.
      residual_fn: (Optional) The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the residual network.
        Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs
        and outputs.
    """
    self._cell = cell
    self._memory_list = [initial_memory,]
    self._out_layer = out_layer
    #self._attention_list = [initial_attention,]
    assert(att_type=='B' or att_type=='L')
    if att_type == 'B':
      self._att_func = _bahdanau_score
    else:
      self._att_func = _luong_score
    self._att_layer = att_layer

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    #inputs = array_ops.concat([inputs, self._attention_list[-1]], 1)
    cell_outputs, new_state = self._cell(inputs, state)
    query = cell_outputs
    memory = self._memory_list[-1]
    expand_query = array_ops.expand_dims(query, 1)
    alignments = self._att_func(query, memory)
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    expanded_attention = math_ops.matmul(expanded_alignments, memory)
    attention = array_ops.squeeze(expanded_attention, [1]) # [batch_size, depth]
    #print('attention', attention, 'concat', array_ops.concat([cell_outputs, attention], 1))
    if self._out_layer is not None:
       new_outputs = self._out_layer(array_ops.concat([cell_outputs, attention], 1))
       #print('new_outputs', new_outputs)
    else:
       new_outputs = cell_outputs
    #self._attention_list.append(attention)
    expand_current_memory = array_ops.expand_dims(array_ops.concat([query, inputs], 1),1)
    #print('expand_current_memory',expand_current_memory)
    if self._att_layer is not None:
       #print('cell_outputs', cell_outputs)
       expand_current_memory = self._att_layer(expand_current_memory)
       #print('expand_current_memory_att',expand_current_memory) 
       #print('query', query)
    new_memory = array_ops.concat([memory, expand_current_memory], 1)   # [batch, time+1, depth]
    self._memory_list.append(new_memory)
    
    return (new_outputs, new_state)