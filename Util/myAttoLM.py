import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.python.layers import core as core_layers
from myAttWrapper import SelfAttOtWrapper
import tensorflow as tf
import numpy as np
import time
import myResidualCell
#import jieba
from bleu import BLEU
import random
import pickle as cPickle
import matplotlib.pyplot as plt


class LM:
    def __init__(self, dp, rnn_size, n_layers, decoder_embedding_dim, max_infer_length, is_jieba, 
                 sess, att_type='B', lr=0.001, grad_clip=5.0, beam_width=5, force_teaching_ratio=1.0, beam_penalty=1.0,
                residual=False, output_keep_prob=0.5, input_keep_prob=0.9, cell_type='lstm', reverse=False, is_save=True,
                decay_scheme='luong234'):
        
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.is_jieba = is_jieba
        self.grad_clip = grad_clip
        self.dp = dp
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
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = 15)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        
    # end constructor

    def build_graph(self):
        self.register_symbols()
        self.add_input_layer()
        with tf.variable_scope('decode'):
            self.add_decoder_for_training()
        with tf.variable_scope('decode', reuse=True):
            self.add_decoder_for_prefix_inference()
        self.add_backward_path()
    # end method

    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None], name="X")
        self.X_seq_len = tf.placeholder(tf.int32, [None], name="X_seq_len")
        self.input_keep_prob = tf.placeholder(tf.float32,name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(tf.float32,name="output_keep_prob")
        self.batch_size = tf.shape(self.X)[0]
        self.init_memory = tf.zeros([self.batch_size, 1, self.rnn_size])
        #self.init_attention = tf.zeros([self.batch_size, self.rnn_size])
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
        self.decoder_cell = SelfAttOtWrapper(self.decoder_cell, self.init_memory, 
                                             att_layer = core_layers.Dense(self.rnn_size, name='att_dense'), 
                                             out_layer = core_layers.Dense(self.rnn_size, name='out_dense'), 
                                             att_type=self.att_type)
        decoder_embedding = tf.get_variable('word_embedding', [len(self.dp.X_w2id), self.decoder_embedding_dim],
                                             tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
        training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs = tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input()),
            sequence_length = self.X_seq_len,
            embedding = decoder_embedding,
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

     
    def add_decoder_for_prefix_inference(self):
        self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.single_cell() for _ in range(1 * self.n_layers)])
        #self.init_attention_tiled = tf.contrib.seq2seq.tile_batch(self.init_attention, self.beam_width)
        self.init_memory_tiled = tf.contrib.seq2seq.tile_batch(self.init_memory, self.beam_width)
        
        self.decoder_cell = SelfAttOtWrapper(self.decoder_cell, 
                                           self.init_memory_tiled, 
                                           att_layer = core_layers.Dense(self.rnn_size, name='att_dense', _reuse=True),
                                           out_layer = core_layers.Dense(self.rnn_size, name='out_dense', _reuse=True),
                                           att_type=self.att_type)
        self.beam_init_state = tf.contrib.seq2seq.tile_batch(self.init_prefix_state, self.beam_width)
        my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell = self.decoder_cell,
            embedding = tf.get_variable('word_embedding'),
            start_tokens = tf.tile(tf.constant([self._x_go], dtype=tf.int32), [self.batch_size]),
            end_token = self._x_eos,
            initial_state = self.beam_init_state,
            beam_width = self.beam_width,
            output_layer = core_layers.Dense(len(self.dp.X_w2id), name='output_dense', _reuse=True),
            length_penalty_weight = self.beam_penalty)
        
        self.prefix_go = tf.placeholder(tf.int32, [None])
        prefix_go_beam = tf.tile(tf.expand_dims(self.prefix_go, 1), [1, self.beam_width])
        prefix_emb = tf.nn.embedding_lookup(tf.get_variable('word_embedding'), prefix_go_beam)
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
        self.batch_loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.X,
                                                     weights = masks,
                                                     average_across_batch=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.learning_rate = tf.constant(self.lr)
        self.learning_rate = self.get_learning_rate_decay(self.decay_scheme)  # decay
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

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
            input_index = input_word
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
                output_str = ''.join([self.dp.X_id2w.get(i, '<-1>') for i in ot]) + input_word
            else:
                output_str = input_word+''.join([self.dp.X_id2w.get(i, '<-1>') for i in ot])
            outputs.append(output_str)
        return outputs
        
    def batch_infer(self, input_words, is_show=True):
        #return ["pass"]
        #xx = [char for char in input_index]
        #if self.reverse:
        #    xx = xx[::-1]
        length = [len(xx) for xx in input_words]
        input_indices = [[self.dp.X_w2id.get(char, self._x_unk) for char in s] for s in input_words]
        prefix_go = []
        #print(length)
        for ipt in input_indices:
            prefix_go.append(ipt[-1])
        #print(prefix_go)
        out_indices, scores = self.sess.run([self.prefix_infer_outputs, self.score], {
            self.X: input_indices, self.X_seq_len: length, self.prefix_go: prefix_go, self.input_keep_prob:1,
                                                    self.output_keep_prob:1})
        outputs = []
        for b in range(len(input_indices)):
            eos_id = self.dp.X_w2id['<EOS>']
            ot = out_indices[b,:,0]
            if eos_id in ot:
                ot = ot.tolist()
                ot = ot[:ot.index(eos_id)]
                #if self.reverse:
                #    ot = ot[::-1]
            #if self.reverse:
            #    output_str = ''.join([self.dp.X_id2w.get(i, '<-1>') for i in ot]) + input_words[b]
            #else:
            output_str = input_words[b] +''.join([self.dp.X_id2w.get(i, '<-1>') for i in ot])
            outputs.append(output_str)
        return outputs
    
    def restore(self, path):
        self.saver.restore(self.sess, path)
        print('restore %s success' % path)
        
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