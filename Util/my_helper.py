# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library of helpers for use with SamplingDecoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.seq2seq.python.ops import helper
__all__ = [
    "myHelper",
]

_transpose_batch_time = decoder._transpose_batch_time  # pylint: disable=protected-access


def _unstack_ta(inp):
  return tensor_array_ops.TensorArray(
      dtype=inp.dtype, size=array_ops.shape(inp)[0],
      element_shape=inp.get_shape()[1:]).unstack(inp)


class MyHelper(helper.Helper):
    """A helper for first generate given prefix, then generate the reminder by sampling.
    """
    def __init__(self, inputs, sequence_length, end_token, embedding, seed=None, time_major=False, name=None,sample_ids_shape=None, sample_ids_dtype=None):

        with ops.name_scope(name, "MyHelper", [inputs, sequence_length]):
            inputs = ops.convert_to_tensor(inputs, name="inputs")
            self._inputs = inputs
            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)
                
            self._input_tas = nest.map_structure(_unstack_ta, inputs)
            self._sequence_length = ops.convert_to_tensor(
              sequence_length, name="sequence_length")
            
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError(
                    "Expected sequence_length to be a vector, but received shape: %s" %
                    self._sequence_length.get_shape())
            
            if callable(embedding):
                self._embedding_fn = embedding
            else:
                self._embedding_fn = (
                   lambda ids: embedding_ops.embedding_lookup(embedding, ids))
            
            self._seed = seed
            
            self._end_token = ops.convert_to_tensor(
              end_token, dtype=dtypes.int32, name="end_token")
            if self._end_token.get_shape().ndims != 0:
                raise ValueError("end_token must be a scalar")

            
            #self._zero_inputs = nest.map_structure(
            #  lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
            # !!!!
            self._batch_size = array_ops.size(sequence_length)
            self._sample_ids_shape = tensor_shape.TensorShape(sample_ids_shape or [])
            self._sample_ids_dtype = sample_ids_dtype or dtypes.int32


    @property
    def inputs(self):
        return self._inputs

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32
    """
    def initialize(self, name=None):
        with ops.name_scope(name, "TrainingHelperInitialize"):
            finished = array_ops.tile([False], [self._batch_size])
            all_finished = math_ops.reduce_all(finished)
            next_inputs = self._embedding_fn(self._input_tas.read(0))
            return (finished, next_inputs)
    """
    def initialize(self, name=None):
        with ops.name_scope(name, "MyHelperInitialize"):
            finished = math_ops.equal(0, self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            next_inputs = self._embedding_fn(self._input_tas.read(0))
            return (finished, next_inputs)
        
    def sample(self, time, outputs, name=None, **unused_kwargs):
        if not isinstance(outputs, ops.Tensor):
              raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                              type(outputs))
        prefixed = (time+1 >= self._sequence_length)
        all_prefixed = math_ops.reduce_all(prefixed)
        
        sample_id_sampler = categorical.Categorical(logits=outputs)
        sample_ids = control_flow_ops.cond(
              all_prefixed, lambda: sample_id_sampler.sample(seed=self._seed),
              lambda: self._input_tas.read(time+1)) #nest.map_structure(lambda inp: inp.read(time+1), self._input_tas)) #self._input_tas.read(time+1))
             
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with ops.name_scope(name, "MyHelperNextInputs",
                            [time, outputs, state]):
            next_time = time + 1
            finished = math_ops.equal(sample_ids, self._end_token)
            prefixed = (next_time >= self._sequence_length)
            all_prefixed = math_ops.reduce_all(prefixed)
            all_finished = math_ops.reduce_all(finished)
            
            next_inputs = self._embedding_fn(sample_ids)
            """
            next_inputs = control_flow_ops.cond(
              all_prefixed, lambda: self._embedding_fn(sample_ids),
              lambda: nest.map_structure(read_from_ta, self._input_tas))
            """
            """
            next_inputs = control_flow_ops.cond(
              all_prefixed, lambda: self._embedding_fn(sample_ids),
              lambda: self._embedding_fn(self._input_tas.read(next_time)))
              #  lambda: self._input_tas.read(next_time))
            """
            return (finished, next_inputs, state)
     