import numpy as np
import time
import copy
import itertools
import random
import pickle as cPickle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.layers import core as core_layers


import os
from Util.myAttWrapper import SelfAttWrapper
from Util import myResidualCell
from Util.bleu import BLEU
from Util.myUtil import *
from Util import my_helper


class LM:
    def __init__(self, dp, rnn_size, n_layers, decoder_embedding_dim, max_infer_length, is_jieba, 
                 sess, qid_list, close_loss_rate=0.0, l2_reg_lambda=0.015, l1_reg_lambda=0.0,  att_type='B', lr=0.001, grad_clip=5.0, beam_width=5, force_teaching_ratio=1.0, beam_penalty=1.0,
                residual=False, output_keep_prob=0.5, input_keep_prob=0.9, cell_type='lstm', reverse=False, is_save=True,
                decay_scheme='luong234'):
        
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.is_jieba = is_jieba
        self.grad_clip = grad_clip
        self.dp = dp
        self.qid_list = qid_list
        self.l2_reg_lambda = l2_reg_lambda
        self.l1_reg_lambda = l1_reg_lambda
        self.decoder_embedding_dim = decoder_embedding_dim
        self.beam_width = beam_width
        self.beam_penalty = beam_penalty
        self.max_infer_length = max_infer_length
        self.residual = residual
        self.decay_scheme = decay_scheme
        if self.residual:
            assert decoder_embedding_dim == rnn_size
        self.reverse = reverse
        self.cell_type = cell_type
        self.force_teaching_ratio = force_teaching_ratio
        self._output_keep_prob = output_keep_prob
        self._input_keep_prob = input_keep_prob
        self.is_save = is_save
        self.sess = sess
        self.att_type = att_type
        self.lr=lr
        self.close_loss_rate = close_loss_rate
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.opt_var = [x for x in tf.global_variables() if 'Ftrl' in x.name or 'Momentum' in x.name]
        #print(len(tf.global_variables()), len([x for x in tf.global_variables() if x != self.extra_embedding]))
        self.saver = tf.train.Saver([x for x in tf.trainable_variables() if x not in self.extra_embedding_list], max_to_keep = 15)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        
    # end constructor

    def build_graph(self):
        self.register_symbols()
        self.add_input_layer()
        with tf.variable_scope('decode'):
            self.add_decoder_for_training()
        with tf.variable_scope('decode', reuse=True):
            self.add_decoder_for_prefix_inference()
        with tf.variable_scope('decode', reuse=True):
            self.add_decoder_for_sample()
        with tf.variable_scope('decode', reuse=True):
            self.add_decoder_for_prefix_sample()
        self.build_embop()
        self.build_l2_distance()
        self.build_projection()
        self.add_assign()
        self.build_nearst()
        self.add_backward_path()
    # end method
    
    def add_assign(self):
        self.assgin_placeholder_list = []
        for i in range(len(self.qid_list)):
            self.assgin_placeholder_list.append(tf.placeholder(tf.float32, [1, self.decoder_embedding_dim], name='assgin_placeholder_%d' % i))
        self.assign_op_list = []
        for i in range(len(self.qid_list)):
            self.assign_op_list.append(self.extra_embedding_list[i].assign(self.assgin_placeholder_list[i]))
            
    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None], name="X")
        self.Y = tf.placeholder(tf.int32, [None, None], name="Y")
        self.X_seq_len = tf.placeholder(tf.int32, [None], name="X_seq_len")
        self.Y_seq_len = tf.placeholder(tf.int32, [None], name="Y_seq_len")
        self.input_keep_prob = tf.placeholder(tf.float32,name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(tf.float32,name="output_keep_prob")
        self.batch_size = tf.shape(self.X)[0]
        self.init_memory = tf.zeros([self.batch_size, 1, self.rnn_size])
        self.init_attention = tf.zeros([self.batch_size, self.rnn_size])
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
    # end method

    def single_cell(self, reuse=False):
        if self.cell_type == 'lstm':
             cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_size, reuse=reuse)
        else:
            cell = tf.contrib.rnn.GRUBlockCell(self.rnn_size)    
        cell = tf.contrib.rnn.DropoutWrapper(cell, self.output_keep_prob, self.input_keep_prob)
        if self.residual:
            cell = myResidualCell.ResidualWrapper(cell)
        return cell

    def processed_decoder_input(self):
        main = tf.strided_slice(self.X, [0, 0], [self.batch_size, -1], [1, 1]) # remove last char
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self._x_go), main], 1)
        return decoder_input

    def add_decoder_for_training(self):
        self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.single_cell() for _ in range(1 * self.n_layers)])
        self.decoder_cell = SelfAttWrapper(self.decoder_cell, self.init_attention, self.init_memory, att_layer = core_layers.Dense(self.rnn_size, name='att_dense'), att_type=self.att_type)
        self.decoder_embedding = tf.get_variable('word_embedding', [len(self.dp.X_w2id), self.decoder_embedding_dim],
                                             tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
        #print(decoder_embedding)
        #print(decoder_embedding[:1], decoder_embedding[2:])
        self.extra_embedding_list = []
        for i in self.qid_list:
            self.extra_embedding_list.append(tf.get_variable('extra_embedding_%d' % i, [1, self.decoder_embedding_dim],
                                                 tf.float32, tf.random_uniform_initializer(-1.0, 1.0)))
        #print(self.extra_embedding_list)
        self.extra_embeddings = tf.concat(self.extra_embedding_list, axis=0)
        for i,extra_embedding in enumerate(self.extra_embedding_list):
            self.decoder_embedding = tf.concat([self.decoder_embedding[:self.qid_list[i]], extra_embedding, self.decoder_embedding[self.qid_list[i]+1:]], axis=0)
        #print(self.decoder_embedding)
        training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs = tf.nn.embedding_lookup(self.decoder_embedding, self.processed_decoder_input()),
            sequence_length = self.X_seq_len,
            embedding = self.decoder_embedding,
            sampling_probability = 1 - self.force_teaching_ratio,
            time_major = False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = self.decoder_cell,
            helper = training_helper,
            initial_state = self.decoder_cell.zero_state(self.batch_size, tf.float32), #.clone(cell_state=self.encoder_state),
            output_layer = core_layers.Dense(len(self.dp.X_w2id), name='output_dense'))
        training_decoder_output, training_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = training_decoder,
            impute_finished = True,
            maximum_iterations = tf.reduce_max(self.X_seq_len))
        self.training_logits = training_decoder_output.rnn_output
        self.init_prefix_state = training_final_state
        self.output_prob = tf.nn.softmax(self.training_logits, -1)

    def add_decoder_for_sample(self):
        self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.single_cell() for _ in range(1 * self.n_layers)])
        self.decoder_cell = SelfAttWrapper(self.decoder_cell, self.init_attention, self.init_memory, att_layer = core_layers.Dense(self.rnn_size, name='att_dense', _reuse=True), att_type=self.att_type)
        word_embedding = tf.get_variable('word_embedding')
        sample_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
            embedding= word_embedding, 
            start_tokens = tf.tile(tf.constant([self._x_go], dtype=tf.int32), [self.batch_size]), 
            end_token = self._x_eos)
        sample_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = self.decoder_cell,
            helper = sample_helper,
            initial_state = self.decoder_cell.zero_state(self.batch_size, tf.float32),#.clone(cell_state=self.encoder_state),
            output_layer = core_layers.Dense(len(self.dp.X_w2id),name='output_dense', _reuse=True))
        sample_decoder_output, self.sample_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = sample_decoder,
            impute_finished = False,
            maximum_iterations = self.max_infer_length)
        self.sample_output = sample_decoder_output.sample_id
        
    def add_decoder_for_prefix_sample(self):
        self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.single_cell() for _ in range(1 * self.n_layers)])
        self.decoder_cell = SelfAttWrapper(self.decoder_cell, self.init_attention, self.init_memory, att_layer = core_layers.Dense(self.rnn_size, name='att_dense', _reuse=True), att_type=self.att_type)
        word_embedding = tf.get_variable('word_embedding')
        prefix_sample_helper = my_helper.MyHelper(
            inputs = self.processed_decoder_input(),
            sequence_length = self.X_seq_len,
            embedding= word_embedding, 
            end_token = self._x_eos)
        sample_prefix_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = self.decoder_cell,
            helper = prefix_sample_helper,
            initial_state = self.decoder_cell.zero_state(self.batch_size, tf.float32),#.clone(cell_state=self.encoder_state),
            output_layer = core_layers.Dense(len(self.dp.X_w2id), name='output_dense', _reuse=True))
        sample_decoder_prefix_output, self.sample_prefix_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = sample_prefix_decoder,
            impute_finished = False,
            maximum_iterations = self.max_infer_length)
        self.sample_prefix_output = sample_decoder_prefix_output.sample_id
        
    def add_decoder_for_prefix_inference(self):
        self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.single_cell() for _ in range(1 * self.n_layers)])
        self.init_attention_tiled = tf.contrib.seq2seq.tile_batch(self.init_attention, self.beam_width)
        self.init_memory_tiled = tf.contrib.seq2seq.tile_batch(self.init_memory, self.beam_width)
        
        self.decoder_cell = SelfAttWrapper(self.decoder_cell, self.init_attention_tiled, self.init_memory_tiled, att_layer = core_layers.Dense(self.rnn_size, name='att_dense', _reuse=True),att_type=self.att_type)
        self.beam_init_state = tf.contrib.seq2seq.tile_batch(self.init_prefix_state, self.beam_width)
        my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell = self.decoder_cell,
            embedding = self.decoder_embedding,
            start_tokens = tf.tile(tf.constant([self._x_go], dtype=tf.int32), [self.batch_size]),
            end_token = self._x_eos,
            initial_state = self.beam_init_state,
            beam_width = self.beam_width,
            output_layer = core_layers.Dense(len(self.dp.X_w2id), name='output_dense', _reuse=True),
            length_penalty_weight = self.beam_penalty)
        
        self.prefix_go = tf.placeholder(tf.int32, [None])
        prefix_go_beam = tf.tile(tf.expand_dims(self.prefix_go, 1), [1, self.beam_width])
        prefix_emb = tf.nn.embedding_lookup(self.decoder_embedding, prefix_go_beam)
        my_decoder._start_inputs = prefix_emb
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = my_decoder,
            impute_finished = False,
            maximum_iterations = self.max_infer_length)
        self.prefix_infer_outputs = predicting_decoder_output.predicted_ids
        self.score = predicting_decoder_output.beam_search_decoder_output.scores
        
    
            
    def add_backward_path(self):
        masks = tf.sequence_mask(self.X_seq_len, tf.reduce_max(self.X_seq_len), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.X,
                                                     weights = masks)
        
        l1_regularizer = tf.contrib.layers.l1_regularizer(self.l1_reg_lambda)
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg_lambda)
        self.l1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, self.extra_embedding_list)
        self.l2_loss = tf.contrib.layers.apply_regularization(l2_regularizer, self.extra_embedding_list)
        #print(tf.norm(self.nearest_emb_placeholder-self.extra_embeddings, ord='euclidean'))
        self.close_loss = tf.reduce_sum(tf.norm(self.nearest_emb_placeholder-self.extra_embeddings, ord='euclidean')) * self.close_loss_rate
        if self.close_loss_rate > 0.0:
            self.update_loss = self.close_loss + self.l2_loss + self.l1_loss + tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                                                                targets = self.Y,
                                                                                                weights = masks) 
        else:
            self.update_loss = self.l2_loss + self.l1_loss + tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                                                                targets = self.Y,
                                                                                                weights = masks) 
        self.update_batch_loss = self.l2_loss + self.l1_loss + tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                                                                targets = self.Y,
                                                                                                weights = masks,
                                                                                                average_across_batch=False) 
        
        self.batch_loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.X,
                                                     weights = masks,
                                                     average_across_batch=False)
        self.time_batch_loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.X,
                                                     weights = masks,
                                                     average_across_batch=False,
                                                     average_across_timesteps=False)
        self.time_loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.X,
                                                     weights = masks,
                                                     average_across_timesteps=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        
        update_params = self.extra_embedding_list
        self.Dgrad = tf.gradients(self.update_loss, update_params)
        self.Dgrad_list = [tf.gradients(self.update_loss, [update_params[k]]) for k in range(len(update_params))]
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.lbfgs_op = tf.contrib.opt.ScipyOptimizerInterface(self.update_loss, var_list=update_params, method='L-BFGS-B', options={'maxiter': 100,'disp': 0})
        #update_clipped_gradients, _ = tf.clip_by_global_norm(self.Dgrad, 5.0)
        #print(self.lbfgs_op)
        
        self.learning_rate = tf.constant(self.lr)
        self.learning_rate = self.get_learning_rate_decay(self.decay_scheme)  # decay
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        self.update_op = dict()
        lr = self.lr
        self.update_op['Adam'] = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.Dgrad, update_params))
        self.update_op['Adadelta'] = tf.train.AdadeltaOptimizer(lr).apply_gradients(zip(self.Dgrad, update_params))
        self.update_op['Adagrad'] = tf.train.AdagradOptimizer(lr).apply_gradients(zip(self.Dgrad, update_params))
        self.update_op['GradientDescent'] = tf.train.GradientDescentOptimizer(lr).apply_gradients(zip(self.Dgrad, update_params))
        self.update_op['Momentum'] = tf.train.MomentumOptimizer(lr, 0.9).apply_gradients(zip(self.Dgrad, update_params))
        self.update_op['Nesterov'] = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).apply_gradients(zip(self.Dgrad, update_params))
        self.update_op['Ftrl'] = tf.train.FtrlOptimizer(lr).apply_gradients(zip(self.Dgrad, update_params))
        self.update_op['ProximalAdagrad'] = tf.train.ProximalAdagradOptimizer(lr).apply_gradients(zip(self.Dgrad, update_params))
        self.update_op['ProximalGradientDescent'] = tf.train.ProximalGradientDescentOptimizer(lr).apply_gradients(zip(self.Dgrad, update_params))
        self.update_op['RMSProp'] = tf.train.RMSPropOptimizer(lr).apply_gradients(zip(self.Dgrad, update_params))
        for k,extra_emb in enumerate(self.extra_embedding_list):
            self.update_op['Ftrl_%d' % k] = tf.train.FtrlOptimizer(lr).apply_gradients(zip(self.Dgrad_list[k], [update_params[k]]))
            self.update_op['Nesterov_%d' % k] = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).apply_gradients(zip(self.Dgrad_list[k], [update_params[k]]))
            self.update_op['GradientDescent_%d' % k] = tf.train.GradientDescentOptimizer(lr).apply_gradients(zip(self.Dgrad_list[k], [update_params[k]]))
            self.update_op['Momentum_%d' % k] = tf.train.MomentumOptimizer(lr,0.9).apply_gradients(zip(self.Dgrad_list[k], [update_params[k]]))
            self.update_op['Adam_%d' % k] = tf.train.AdamOptimizer(lr).apply_gradients(zip(self.Dgrad_list[k], [update_params[k]]))

        
    def build_nearst(self):
        self.nearest_emb_placeholder = tf.placeholder(tf.float32, [1, self.decoder_embedding_dim], name='nearest_emb')
                                  
    def build_embop(self):
        nemb = tf.nn.l2_normalize(self.decoder_embedding, 1)
        self.nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(nemb, self.nearby_word)
        
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        #print('cos', nearby_emb,nearby_dist)
        self.nearby_dist = nearby_dist
        self.nearby_val, self.nearby_idx = tf.nn.top_k(nearby_dist, len(self.dp.X_w2id))
    
    def build_l2_distance(self):
        nemb = self.decoder_embedding
        nearby_emb = tf.gather(nemb, self.nearby_word)
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(nearby_emb-nemb), 1))
        self.eu_nearby_val, self.eu_nearby_idx = tf.nn.top_k(-euclidean, len(self.dp.X_w2id))
        
    def build_projection(self):
        nemb = self.decoder_embedding
        nearby_emb = tf.gather(nemb, self.nearby_word)
        z_x = (nemb - nearby_emb)
        self.g = tf.placeholder(tf.float32, [1, self.decoder_embedding_dim], name='g')
        g_norm = tf.nn.l2_normalize(self.g)
        #print(z_x, g_norm)
        self.proj = tf.matmul(g_norm, z_x, transpose_b=True)
        #print(self.proj)
        self.nearby_pro_val, self.nearby_pro_idx = tf.nn.top_k(self.proj, len(self.dp.X_w2id))
    
    def find_nearnest(self, idx, topk=10):
        nearby_val, nearby_idx = self.sess.run([self.nearby_val, self.nearby_idx], {self.nearby_word:idx})
        return nearby_val[:topk], nearby_idx[:topk]
    
    def register_symbols(self):
        self._x_go = self.dp.X_w2id['<GO>']
        self._x_eos = self.dp.X_w2id['<EOS>']
        self._x_pad = self.dp.X_w2id['<PAD>']
        self._x_unk = self.dp.X_w2id['<UNK>']
        
        
    def infer(self, input_word, batch_size=1, is_show=True):
        #return ["pass"]
        if self.is_jieba:
            input_index = list(jieba.cut(input_word))
        else:
            input_index = input_word.split(' ')
        xx = [char for char in input_index]
        if self.reverse:
            xx = xx[::-1]
        length = [len(xx),] * batch_size
        input_indices = [[self.dp.X_w2id.get(char, self._x_unk) for char in xx]] * batch_size
        prefix_go = []
        for ipt in input_indices:
            prefix_go.append(ipt[-1])
        out_indices, scores = self.sess.run([self.prefix_infer_outputs, self.score], {
            self.X: input_indices, self.X_seq_len: length, self.prefix_go: prefix_go, self.input_keep_prob:1,
                                                    self.output_keep_prob:1})
        outputs = []
        for idx in range(out_indices.shape[-1]):
            eos_id = self.dp.X_w2id['<EOS>']
            ot = out_indices[0,:,idx]
            if eos_id in ot:
                ot = ot.tolist()
                ot = ot[:ot.index(eos_id)]
                if self.reverse:
                    ot = ot[::-1]
            if self.reverse:
                output_str = ' '.join([self.dp.X_id2w.get(i, '<-1>') for i in ot]) + ' '+ input_word
            else:
                output_str = input_word+' ' + ' '.join([self.dp.X_id2w.get(i, '<-1>') for i in ot])
            outputs.append(output_str)
        return outputs
    
    def infer_with_scores(self, input_word, batch_size=1, is_show=True):
        #return ["pass"]
        if self.is_jieba:
            input_index = list(jieba.cut(input_word))
        else:
            input_index = input_word.split(' ')
        xx = [char for char in input_index]
        if self.reverse:
            xx = xx[::-1]
        length = [len(xx),] * batch_size
        input_indices = [[self.dp.X_w2id.get(char, self._x_unk) for char in xx]] * batch_size
        prefix_go = []
        for ipt in input_indices:
            prefix_go.append(ipt[-1])
        out_indices, scores = self.sess.run([self.prefix_infer_outputs, self.score], {
            self.X: input_indices, self.X_seq_len: length, self.prefix_go: prefix_go, self.input_keep_prob:1,
                                                    self.output_keep_prob:1})
        outputs = []
        for idx in range(out_indices.shape[-1]):
            eos_id = self.dp.X_w2id['<EOS>']
            ot = out_indices[0,:,idx]
            if eos_id in ot:
                ot = ot.tolist()
                ot = ot[:ot.index(eos_id)]
                if self.reverse:
                    ot = ot[::-1]
            if self.reverse:
                output_str = ' '.join([self.dp.X_id2w.get(i, '<-1>') for i in ot]) + ' '+ input_word
            else:
                output_str = input_word+' ' + ' '.join([self.dp.X_id2w.get(i, '<-1>') for i in ot])
            outputs.append(output_str)
        return outputs, scores
        
    def generate(self, batch_size=1, is_show=True):
        fake_x = [[1] for _ in range(batch_size)]
        out_indices = self.sess.run(self.sample_output, {self.X: fake_x, self.input_keep_prob:1, self.output_keep_prob:1})
        #print(out_indices.shape)
        outputs = []
        for ot in out_indices:
            eos_id = self.dp.X_w2id['<EOS>']
            if eos_id in ot:
                ot = ot.tolist()
                ot = ot[:ot.index(eos_id)]
                if self.reverse:
                    ot = ot[::-1]
            if self.reverse:
                output_str = ' '.join([self.dp.X_id2w.get(i, '<-1>') for i in ot])
            else:
                output_str = ' '.join([self.dp.X_id2w.get(i, '<-1>') for i in ot])
            outputs.append(output_str)
        return out_indices, outputs
    
    def rollout(self, input_word, batch_size=1, is_show=True):
        if self.is_jieba:
            input_index = list(jieba.cut(input_word))
        else:
            input_index = input_word
        xx = [char for char in input_index]
        if self.reverse:
            xx = xx[::-1]
        length = [len(xx)+1] * batch_size
        input_indices = [[self.dp.X_w2id.get(char, self._x_unk) for char in xx]] * batch_size
        input_indices = [x+[self.dp.X_w2id['<EOS>'],] for x in input_indices]
        #print(input_indices)
        out_indices = self.sess.run(self.sample_prefix_output, {
            self.X: input_indices, self.X_seq_len: length, self.input_keep_prob:1,
                                                    self.output_keep_prob:1})
        outputs = []
        for ot in out_indices:
            eos_id = self.dp.X_w2id['<EOS>']
            if eos_id in ot:
                ot = ot.tolist()
                ot = ot[:ot.index(eos_id)]
                if self.reverse:
                    ot = ot[::-1]
            
            if self.reverse:
                output_str = ' '.join([self.dp.X_id2w.get(i, '<-1>') for i in ot])
            else:
                output_str = ' '.join([self.dp.X_id2w.get(i, '<-1>') for i in ot])
            outputs.append(output_str)
        return outputs
    
    def restore(self, path):
        self.saver.restore(self.sess, path)
        #print('restore %s success' % path)
        
    def get_learning_rate_decay(self, decay_scheme='luong234'):
        num_train_steps = self.dp.num_steps
        if decay_scheme == "luong10":
            start_decay_step = int(num_train_steps / 2)
            remain_steps = num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 10)  # decay 10 times
            decay_factor = 0.5
        else:
            start_decay_step = int(num_train_steps * 2 / 3)
            remain_steps = num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 4)  # decay 4 times
            decay_factor = 0.5
        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")
    
    def _opt_init(self):
        self.sess.run(tf.variables_initializer(self.opt_var))
    
    def setup_summary(self):
        train_loss = tf.Variable(0.)
        tf.summary.scalar('Train_loss', train_loss)
        
        test_loss = tf.Variable(0.)
        tf.summary.scalar('Test_loss', test_loss)
        
        bleu_score = tf.Variable(0.)
        tf.summary.scalar('BLEU_score', bleu_score)

        tf.summary.scalar('lr_rate', self.learning_rate)
        
        summary_vars = [train_loss, test_loss, bleu_score]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op