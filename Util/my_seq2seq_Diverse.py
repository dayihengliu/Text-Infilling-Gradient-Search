import os
from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import numpy as np
import time
import myResidualCell
import jieba
from bleu import BLEU
import random
import pickle as cPickle
import matplotlib.pyplot as plt
import DiverseDecode



class Seq2Seq:
    def __init__(self, dp, rnn_size, n_layers, encoder_embedding_dim, decoder_embedding_dim, max_infer_length,
                 sess, gamma, lr=0.001, grad_clip=5.0, beam_width=5, force_teaching_ratio=1.0, beam_penalty=1.0,
                residual=False, output_keep_prob=0.5, input_keep_prob=0.9, is_save=True, cell_type='lstm', reverse=False,
                decay_scheme='luong234'):
        
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.dp = dp
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.beam_width = beam_width
        self.beam_penalty = beam_penalty
        self.max_infer_length = max_infer_length
        self.residual = residual
        self.decay_scheme = decay_scheme
        if self.residual:
            assert encoder_embedding_dim == rnn_size
            assert decoder_embedding_dim == rnn_size
        self.reverse = reverse
        self.cell_type = cell_type
        self.force_teaching_ratio = force_teaching_ratio
        self._output_keep_prob = output_keep_prob
        self._input_keep_prob = input_keep_prob
        self.is_save = is_save
        self.sess = sess
        self.gamma = gamma
        self.lr=lr
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = 15)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        
    # end constructor

    def build_graph(self):
        self.register_symbols()
        self.add_input_layer()
        self.add_encoder_layer()
        with tf.variable_scope('decode'):
            self.add_decoder_for_training()
        with tf.variable_scope('decode', reuse=True):
            self.add_decoder_for_inference()
        with tf.variable_scope('decode', reuse=True):
            self.add_decoder_for_prefix_inference()
        self.add_backward_path()
    # end method
    
    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.
        Which is returned is determined by how the AttentionMechanism(s) were passed
        to the constructor.
        Args:
          seq: A non-empty sequence of items or generator.
        Returns:
           Either the values in the sequence as a tuple if AttentionMechanism(s)
           were passed to the constructor as a sequence or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]
        
    def add_input_layer(self):
        self.X = tf.placeholder(tf.int32, [None, None], name="X")
        self.Y = tf.placeholder(tf.int32, [None, None], name="Y")
        self.X_seq_len = tf.placeholder(tf.int32, [None], name="X_seq_len")
        self.Y_seq_len = tf.placeholder(tf.int32, [None], name="Y_seq_len")
        self.input_keep_prob = tf.placeholder(tf.float32,name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(tf.float32,name="output_keep_prob")
        self.batch_size = tf.shape(self.X)[0]
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
    
    def add_encoder_layer(self):
        encoder_embedding = tf.get_variable('encoder_embedding', [len(self.dp.X_w2id), self.encoder_embedding_dim],
                                             tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
        
        self.encoder_inputs = tf.nn.embedding_lookup(encoder_embedding, self.X)
        bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = tf.contrib.rnn.MultiRNNCell([self.single_cell() for _ in range(self.n_layers)]), 
            cell_bw = tf.contrib.rnn.MultiRNNCell([self.single_cell() for _ in range(self.n_layers)]),
            inputs = self.encoder_inputs,
            sequence_length = self.X_seq_len,
            dtype = tf.float32,
            scope = 'bidirectional_rnn')
        self.encoder_out = tf.concat(bi_encoder_output, 2)
        encoder_state = []
        for layer_id in range(self.n_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
        self.encoder_state = tuple(encoder_state)
    """
    def add_encoder_layer(self):
        encoder_embedding = tf.get_variable('encoder_embedding', [len(self.dp.X_w2id), self.encoder_embedding_dim],
                                             tf.float32, tf.random_uniform_initializer(-1.0, 1.0)) 
        self.encoder_out = tf.nn.embedding_lookup(encoder_embedding, self.X)
        for n in range(self.n_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self.single_cell(), cell_bw = self.single_cell(),
                inputs = self.encoder_out,
                sequence_length = self.X_seq_len,
                dtype = tf.float32,
                scope = 'bidirectional_rnn_'+str(n))
            self.encoder_out = tf.concat((out_fw, out_bw), 2)
        self.encoder_state = ()
        for n in range(self.n_layers): # replicate top-most state
            self.encoder_state += (state_fw, state_bw)
    """
    def processed_decoder_input(self):
        main = tf.strided_slice(self.Y, [0, 0], [self.batch_size, -1], [1, 1]) # remove last char
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self._y_go), main], 1)
        return decoder_input

    def add_attention_for_training(self):
        if self.cell_type == 'lstm':
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units = self.rnn_size, 
                memory = self.encoder_out,
                memory_sequence_length = self.X_seq_len,
                normalize=True)
        else:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units = self.rnn_size, 
                memory = self.encoder_out,
                memory_sequence_length = self.X_seq_len,
                scale=True)
        
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell = tf.nn.rnn_cell.MultiRNNCell([self.single_cell() for _ in range(2 * self.n_layers)]),
            attention_mechanism = attention_mechanism,
            attention_layer_size = self.rnn_size)

    def add_decoder_for_training(self):
        self.add_attention_for_training()
        decoder_embedding = tf.get_variable('decoder_embedding', [len(self.dp.Y_w2id), self.decoder_embedding_dim],
                                             tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
        training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs = tf.nn.embedding_lookup(decoder_embedding, self.processed_decoder_input()),
            sequence_length = self.Y_seq_len,
            embedding = decoder_embedding,
            sampling_probability = 1 - self.force_teaching_ratio,
            time_major = False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = self.decoder_cell,
            helper = training_helper,
            initial_state = self.decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state),
            output_layer = core_layers.Dense(len(self.dp.Y_w2id)))
        training_decoder_output, training_final_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = training_decoder,
            impute_finished = True,
            maximum_iterations = tf.reduce_max(self.Y_seq_len))
        self.training_logits = training_decoder_output.rnn_output
        self.init_prefix_state = training_final_state

    def add_attention_for_inference(self):
        self.encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_out, self.beam_width)
        self.encoder_state_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.beam_width)
        self.X_seq_len_tiled = tf.contrib.seq2seq.tile_batch(self.X_seq_len, self.beam_width)
        if self.cell_type == 'lstm':
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units = self.rnn_size, 
                memory = self.encoder_out_tiled,
                memory_sequence_length = self.X_seq_len_tiled,
                normalize=True)
        else:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units = self.rnn_size, 
                memory = self.encoder_out_tiled,
                memory_sequence_length = self.X_seq_len_tiled,
                scale=True)
        self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell = tf.nn.rnn_cell.MultiRNNCell([self.single_cell(reuse=True) for _ in range(2 * self.n_layers)]),
            attention_mechanism = attention_mechanism,
            attention_layer_size = self.rnn_size)
        self.attention_mechanism = attention_mechanism

    def add_decoder_for_inference(self):
        self.add_attention_for_inference()
        """
        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell = self.decoder_cell,
            embedding = tf.get_variable('decoder_embedding'),
            start_tokens = tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
            end_token = self._y_eos,
            initial_state = self.decoder_cell.zero_state(self.batch_size * self.beam_width, tf.float32).clone(
                            cell_state = self.encoder_state_tiled),
            beam_width = self.beam_width,
            output_layer = core_layers.Dense(len(self.dp.Y_w2id), _reuse=True),
            length_penalty_weight = self.beam_penalty)
        """
        predicting_decoder = DiverseDecode.BeamSearchDecoder(
            cell = self.decoder_cell,
            embedding = tf.get_variable('decoder_embedding'),
            start_tokens = tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
            end_token = self._y_eos,
            gamma = self.gamma,
            initial_state = self.decoder_cell.zero_state(self.batch_size * self.beam_width, tf.float32).clone(cell_state = self.encoder_state_tiled),
            beam_width = self.beam_width,
            vocab_size = len(self.dp.Y_w2id),
            output_layer = core_layers.Dense(len(self.dp.Y_w2id), _reuse=True),
            length_penalty_weight = self.beam_penalty)
        
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = predicting_decoder,
            impute_finished = False,
            maximum_iterations = self.max_infer_length)
        self.predicting_ids = predicting_decoder_output.predicted_ids
        self.score = predicting_decoder_output.beam_search_decoder_output.scores
        
    def add_decoder_for_prefix_inference(self):
        self.add_attention_for_inference()
        prefix_cell_state = tf.contrib.seq2seq.tile_batch(self.init_prefix_state.cell_state, self.beam_width)
        prefix_attention = tf.contrib.seq2seq.tile_batch(self.init_prefix_state.attention, self.beam_width)
        prefix_time = self.init_prefix_state.time
        prefix_alignments = self.init_prefix_state.alignments
        prefix_alignment_history = self.init_prefix_state.alignment_history
        
        init_state = tf.contrib.seq2seq.AttentionWrapperState(cell_state=prefix_cell_state, 
                                                      attention=prefix_attention, time=prefix_time, 
                                                      attention_state=self.init_prefix_state.attention_state,
                                                      alignments=prefix_alignments,
                                                      alignment_history=prefix_alignment_history)
        predicting_decoder = DiverseDecode.BeamSearchDecoder(
            cell = self.decoder_cell,
            embedding = tf.get_variable('decoder_embedding'),
            start_tokens = tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
            end_token = self._y_eos,
            gamma = self.gamma,
            initial_state = init_state,
            beam_width = self.beam_width,
            vocab_size = len(self.dp.Y_w2id),
            output_layer = core_layers.Dense(len(self.dp.Y_w2id), _reuse=True),
            length_penalty_weight = self.beam_penalty)
        """
        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell = self.decoder_cell,
            embedding = tf.get_variable('decoder_embedding'),
            start_tokens = tf.tile(tf.constant([self._y_go], dtype=tf.int32), [self.batch_size]),
            end_token = self._y_eos,
            initial_state = init_state,
            beam_width = self.beam_width,
            output_layer = core_layers.Dense(len(self.dp.Y_w2id), _reuse=True),
            length_penalty_weight = self.beam_penalty)
        """
        self.prefix_go = tf.placeholder(tf.int32, [None])
        prefix_go_beam = tf.tile(tf.expand_dims(self.prefix_go, 1), [1, self.beam_width])
        prefix_emb = tf.nn.embedding_lookup(tf.get_variable('decoder_embedding'), prefix_go_beam)
        predicting_decoder._start_inputs = prefix_emb
        predicting_prefix_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = predicting_decoder,
            impute_finished = False,
            maximum_iterations = self.max_infer_length)
        self.predicting_prefix_ids = predicting_prefix_decoder_output.predicted_ids
        self.prefix_score = predicting_prefix_decoder_output.beam_search_decoder_output.scores

    def add_backward_path(self):
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
                                                     weights = masks)
        self.batch_loss = tf.contrib.seq2seq.sequence_loss(logits = self.training_logits,
                                                     targets = self.Y,
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
        
        self._y_go = self.dp.Y_w2id['<GO>']
        self._y_eos = self.dp.Y_w2id['<EOS>']
        self._y_pad = self.dp.Y_w2id['<PAD>']
        self._y_unk = self.dp.Y_w2id['<UNK>']
    
    def infer(self, input_word):
        #if self.reverse:
        #    input_word = input_word[::-1]
        input_indices = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word]
        out_indices = self.sess.run(self.predicting_ids, {
            self.X: [input_indices], self.X_seq_len: [len(input_indices)], self.output_keep_prob:1, self.input_keep_prob:1})
        outputs = []
        for idx in range(out_indices.shape[-1]):
            eos_id = self.dp.Y_w2id['<EOS>']
            ot = out_indices[0,:,idx]
            if eos_id in ot:
                ot = ot.tolist()
                ot = ot[:ot.index(eos_id)]
            if self.reverse:
                ot = ot[::-1]
            output_str = ''.join([self.dp.Y_id2w.get(i, u'&') for i in ot])
            outputs.append(output_str)
        return outputs
    
    def prefix_infer(self, input_word, prefix):
        input_indices_X = [self.dp.X_w2id.get(char, self._x_unk) for char in input_word]
        input_indices_Y = [self.dp.Y_w2id.get(char, self._y_unk) for char in prefix]
        prefix_go = []
        prefix_go.append(input_indices_Y[-1]) 
        out_indices, scores = self.sess.run([self.predicting_prefix_ids, self.prefix_score], {
            self.X: [input_indices_X], self.X_seq_len: [len(input_indices_X)], self.Y:[input_indices_Y], self.Y_seq_len:[len(input_indices_Y)],
            self.prefix_go: prefix_go, self.input_keep_prob:1, self.output_keep_prob:1})
        
        outputs = []
        for idx in range(out_indices.shape[-1]):
            eos_id = self.dp.Y_w2id['<EOS>']
            ot = out_indices[0,:,idx]
            if eos_id in ot:
                ot = ot.tolist()
                ot = ot[:ot.index(eos_id)]
                if self.reverse:
                    ot = ot[::-1]
            if self.reverse:
                output_str = ''.join([self.dp.Y_id2w.get(i, u'&') for i in ot]) + prefix
            else:
                output_str = prefix + ''.join([self.dp.Y_id2w.get(i, u'&') for i in ot])
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
    
class Seq2Seq_DP:
    def __init__(self, X_indices, Y_indices, X_w2id, Y_w2id, BATCH_SIZE, n_epoch, split_ratio=0.2):
        assert len(X_indices) == len(Y_indices)
        num_test = int(len(X_indices) * split_ratio)
        self.n_epoch = n_epoch
        self.X_train = np.array(X_indices[num_test:])
        self.Y_train = np.array(Y_indices[num_test:])
        self.X_test = np.array(X_indices[:num_test])
        self.Y_test = np.array(Y_indices[:num_test])
        self.num_batch = int(len(self.X_train) / BATCH_SIZE)
        self.num_steps = self.num_batch * self.n_epoch
        self.batch_size = BATCH_SIZE
        self.X_w2id = X_w2id
        self.X_id2w = dict(zip(X_w2id.values(), X_w2id.keys()))
        self.Y_w2id = Y_w2id
        self.Y_id2w = dict(zip(Y_w2id.values(), Y_w2id.keys()))
        self._x_pad = self.X_w2id['<PAD>']
        self._y_pad = self.Y_w2id['<PAD>']
        print('Train_data: %d | Test_data: %d | Batch_size: %d | Num_batch: %d | X_vocab_size: %d | Y_vocab_size: %d' % (len(self.X_train), len(self.X_test), BATCH_SIZE, self.num_batch, len(self.X_w2id), len(self.Y_w2id)))
        
    def next_batch(self, X, Y):
        r = np.random.permutation(len(X))
        X = X[r]
        Y = Y[r]
        for i in range(0, len(X) - len(X) % self.batch_size, self.batch_size):
            X_batch = X[i : i + self.batch_size]
            Y_batch = Y[i : i + self.batch_size]
            padded_X_batch, X_batch_lens = self.pad_sentence_batch(X_batch, self._x_pad)
            padded_Y_batch, Y_batch_lens = self.pad_sentence_batch(Y_batch, self._y_pad)
            yield (np.array(padded_X_batch),
                   np.array(padded_Y_batch),
                   X_batch_lens,
                   Y_batch_lens)
    
    def sample_test_batch(self):
        padded_X_batch, X_batch_lens = self.pad_sentence_batch(self.X_test[: self.batch_size], self._x_pad)
        padded_Y_batch, Y_batch_lens = self.pad_sentence_batch(self.Y_test[: self.batch_size], self._y_pad)
        return np.array(padded_X_batch), np.array(padded_Y_batch), X_batch_lens, Y_batch_lens
        
    def pad_sentence_batch(self, sentence_batch, pad_int):
        padded_seqs = []
        seq_lens = []
        sentence_batch = sentence_batch.tolist()
        max_sentence_len = max([len(sentence) for sentence in sentence_batch])
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs, seq_lens

class Seq2Seq_util:
    def __init__(self, dp, model, display_freq=50):
        self.display_freq = display_freq
        self.dp = dp
        self.model = model
        
    def train(self, epoch):
        avg_loss = 0.0
        tic = time.time()
        X_test_batch, Y_test_batch, X_test_batch_lens, Y_test_batch_lens = self.dp.sample_test_batch()
        for local_step, (X_train_batch, Y_train_batch, X_train_batch_lens, Y_train_batch_lens) in enumerate(
            self.dp.next_batch(self.dp.X_train, self.dp.Y_train)):
            self.model.step, _, loss = self.model.sess.run([self.model.global_step, self.model.train_op, self.model.loss], 
                                          {self.model.X: X_train_batch,
                                           self.model.Y: Y_train_batch,
                                           self.model.X_seq_len: X_train_batch_lens,
                                           self.model.Y_seq_len: Y_train_batch_lens,
                                           self.model.output_keep_prob:self.model._output_keep_prob,
                                           self.model.input_keep_prob:self.model._input_keep_prob})
            avg_loss += loss
            """
            stats = [loss]
            for i in xrange(len(stats)):
                self.model.sess.run(self.model.update_ops[i], feed_dict={
                    self.model.summary_placeholders[i]: float(stats[i])
                })
            summary_str = self.model.sess.run([self.model.summary_op])
            self.summary_writer.add_summary(summary_str, self.model.step + 1)
            """
            if local_step % (self.dp.num_batch / self.display_freq) == 0:
                val_loss = self.model.sess.run(self.model.loss, {self.model.X: X_test_batch,
                                                     self.model.Y: Y_test_batch,
                                                     self.model.X_seq_len: X_test_batch_lens,
                                                     self.model.Y_seq_len: Y_test_batch_lens,
                                                     self.model.output_keep_prob:1,
                                                     self.model.input_keep_prob:1})
                print("Epoch %d/%d | Batch %d/%d | Train_loss: %.3f | Test_loss: %.3f | Time_cost:%.3f" % (epoch, self.n_epoch, local_step, self.dp.num_batch, avg_loss / (local_step + 1), val_loss, time.time()-tic))
                self.cal()
                tic = time.time()
        return avg_loss / self.dp.num_batch
    
    def test(self):
        avg_loss = 0.0
        for local_step, (X_test_batch, Y_test_batch, X_test_batch_lens, Y_test_batch_lens) in enumerate(
            self.dp.next_batch(self.dp.X_test, self.dp.Y_test)):
            val_loss = self.model.sess.run(self.model.loss, {self.model.X: X_test_batch,
                                                 self.model.Y: Y_test_batch,
                                                 self.model.X_seq_len: X_test_batch_lens,
                                                 self.model.Y_seq_len: Y_test_batch_lens,
                                                 self.model.output_keep_prob:1,
                                                 self.model.input_keep_prob:1})
            avg_loss += val_loss
        return avg_loss / (local_step + 1)
    
    def fit(self, train_dir, is_bleu):
        self.n_epoch = self.dp.n_epoch
        test_loss_list = []
        train_loss_list = []
        time_cost_list = []
        bleu_list = []
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(train_dir, "runs", timestamp))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("Writing to %s" % out_dir)
        checkpoint_prefix = os.path.join(out_dir, "model")
        self.summary_writer = tf.summary.FileWriter(os.path.join(out_dir, 'Summary'), self.model.sess.graph)
        for epoch in range(1, self.n_epoch+1):
            tic = time.time()
            train_loss = self.train(epoch)
            train_loss_list.append(train_loss)
            test_loss = self.test()
            test_loss_list.append(test_loss)
            toc = time.time()
            time_cost_list.append((toc - tic))
            if is_bleu:
                bleu = self.test_bleu()
                bleu_list.append(bleu)
                print("Epoch %d/%d | Train_loss: %.3f | Test_loss: %.3f | Bleu: %.3f" % (epoch, self.n_epoch, train_loss, test_loss, bleu))
            else:
                bleu = 0.0
                print("Epoch %d/%d | Train_loss: %.3f | Test_loss: %.3f" % (epoch, self.n_epoch, train_loss, test_loss))
            
            stats = [train_loss, test_loss, bleu]
            for i in range(len(stats)):
                self.model.sess.run(self.model.update_ops[i], feed_dict={
                    self.model.summary_placeholders[i]: float(stats[i])
                })
            summary_str = self.model.sess.run(self.model.summary_op)
            self.summary_writer.add_summary(summary_str, epoch)
            if self.model.is_save:
                cPickle.dump((train_loss_list, test_loss_list, time_cost_list, bleu_list), open(os.path.join(out_dir,"res.pkl"),'wb'))
                path = self.model.saver.save(self.model.sess, checkpoint_prefix, global_step=epoch)
                print("Saved model checkpoint to %s" % path)
    
    def show(self, sent, id2w):
        return "".join([id2w.get(idx, u'&') for idx in sent])
    
    def cal(self, n_example=5):
        train_n_example = int(n_example / 2)
        test_n_example = n_example - train_n_example
        test_idx = [i for i in range(len(self.dp.X_test))]
        train_idx = [i for i in range(len(self.dp.X_train))]
        for _ in random.sample(test_idx, test_n_example):
            example = self.show(self.dp.X_test[_], self.dp.X_id2w)
            y = self.show(self.dp.Y_test[_], self.dp.Y_id2w)
            o = self.model.infer(example)[0]
            print('Input: %s | Output: %s | GroundTruth: %s' % (example, o, y))
        for _ in random.sample(train_idx, train_n_example):
            example = self.show(self.dp.X_train[_], self.dp.X_id2w)
            y = self.show(self.dp.Y_train[_], self.dp.Y_id2w)
            o = self.model.infer(example)[0]
            print('Input: %s | Output: %s | GroundTruth: %s' % (example, o, y)) 
        print("")
        
    def test_bleu(self, N=300, gram=4):
        all_score = []
        for i in range(N):
            input_indices = self.show(self.dp.X_test[i], self.dp.X_id2w)
            o = self.model.infer(input_indices)[0]
            refer4bleu = [[' '.join([self.dp.Y_id2w.get(w, u'&') for w in self.dp.Y_test[i]])]]
            candi = [' '.join(w for w in o)]
            score = BLEU(candi, refer4bleu, gram=gram)
            all_score.append(score)
        return np.mean(all_score)
    
    def show_res(self, path):
        res = cPickle.load(open(path))
        plt.figure(1)
        plt.title('The results') 
        l1, = plt.plot(res[0], 'g')
        l2, = plt.plot(res[1], 'r')
        l3, = plt.plot(res[3], 'b')
        plt.legend(handles = [l1, l2, l3], labels = ["Train_loss","Test_loss","BLEU"], loc = 'best')
        plt.show()
        
    def test_all(self, path, epoch_range, is_bleu=True):
        val_loss_list = []
        bleu_list = []
        for i in range(epoch_range[0], epoch_range[-1]):
            self.model.restore(path + str(i))
            val_loss = self.test()
            val_loss_list.append(val_loss)
            if is_bleu:
                bleu_score = self.test_bleu()
                bleu_list.append(bleu_score)
        plt.figure(1)
        plt.title('The results') 
        l1, = plt.plot(val_loss_list,'r')
        l2, = plt.plot(bleu_list,'b')
        plt.legend(handles = [l1, l2], labels = ["Test_loss","BLEU"], loc = 'best')
        plt.show()
        
    
