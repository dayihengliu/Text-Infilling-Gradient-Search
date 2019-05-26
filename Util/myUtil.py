import numpy as np
import tensorflow as tf
import random
import time
import os
import pickle as cPickle

class LM_DP:
    def __init__(self, X_indices, X_w2id, BATCH_SIZE, n_epoch):
        num_test = int(len(X_indices) * 0.1)
        self.n_epoch = n_epoch
        self.X_train = np.array(X_indices[num_test:])
        self.X_test = np.array(X_indices[:num_test])
        self.num_batch = int(len(self.X_train) / BATCH_SIZE)
        self.num_steps = self.num_batch * self.n_epoch
        self.batch_size = BATCH_SIZE
        self.X_w2id = X_w2id
        self.X_id2w = dict(zip(X_w2id.values(), X_w2id.keys()))
        self._x_pad = self.X_w2id['<PAD>']
        print('Train_data: %d | Test_data: %d | Batch_size: %d | Num_batch: %d | X_vocab_size: %d ' % (len(self.X_train), len(self.X_test), BATCH_SIZE, self.num_batch, len(self.X_w2id)))
        
    def next_batch(self, X):
        r = np.random.permutation(len(X))
        X = X[r]
        for i in range(0, len(X) - len(X) % self.batch_size, self.batch_size):
            X_batch = X[i : i + self.batch_size]
            padded_X_batch, X_batch_lens = self.pad_sentence_batch(X_batch, self._x_pad)
            yield (np.array(padded_X_batch),
                   X_batch_lens)
    
    def sample_test_batch(self):
        padded_X_batch, X_batch_lens = self.pad_sentence_batch(self.X_test[: self.batch_size], self._x_pad)
        return np.array(padded_X_batch), X_batch_lens
        
    def pad_sentence_batch(self, sentence_batch, pad_int):
        padded_seqs = []
        seq_lens = []
        max_sentence_len = max([len(sentence) for sentence in sentence_batch])
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs, seq_lens

    
class LM_util:
    def __init__(self, dp, model, display_freq=3):
        self.display_freq = display_freq
        self.dp = dp
        self.model = model
        
    def train(self, epoch):
        avg_loss = 0.0
        tic = time.time()
        X_test_batch, X_test_batch_lens = self.dp.sample_test_batch()
        for local_step, (X_train_batch, X_train_batch_lens) in enumerate(
            self.dp.next_batch(self.dp.X_train)):
            self.model.step, _, loss = self.model.sess.run([self.model.global_step, self.model.train_op, self.model.loss], 
                                          {self.model.X: X_train_batch,
                                           self.model.X_seq_len: X_train_batch_lens,
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
                                                     self.model.X_seq_len: X_test_batch_lens,
                                                     self.model.output_keep_prob:1,
                                                     self.model.input_keep_prob:1})
                print("Epoch %d/%d | Batch %d/%d | Train_loss: %.3f | Test_loss: %.3f | Time_cost:%.3f" % (epoch, self.n_epoch, local_step, self.dp.num_batch, avg_loss / (local_step + 1), val_loss, time.time()-tic))
                self.cal()
                tic = time.time()
        return avg_loss / self.dp.num_batch
    
    def test(self):
        avg_loss = 0.0
        local_step = 0
        for local_step, (X_test_batch, X_test_batch_lens) in enumerate(
            self.dp.next_batch(self.dp.X_test)):
            val_loss = self.model.sess.run(self.model.loss, {self.model.X: X_test_batch,
                                                 self.model.X_seq_len: X_test_batch_lens,
                                                 self.model.output_keep_prob:1,
                                                 self.model.input_keep_prob:1})
            avg_loss += val_loss
        return avg_loss / (local_step + 1)
    
    def fit(self, train_dir, is_bleu, init_epoch=0):
        self.n_epoch = self.dp.n_epoch
        test_loss_list = []
        train_loss_list = []
        time_cost_list = []
        bleu_list = []
        #timestamp = str(int(time.time()))
        #out_dir = os.path.abspath(os.path.join(train_dir, "runs", timestamp))
        out_dir = train_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("Writing to %s" % out_dir)
        checkpoint_prefix = os.path.join(out_dir, "model")
        self.summary_writer = tf.summary.FileWriter(os.path.join(out_dir, 'Summary'), self.model.sess.graph)
        for epoch in range(init_epoch, init_epoch+self.n_epoch+1):
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
            
            print('============================================')
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
        if self.model.reverse:
            return " ".join([id2w.get(idx, u'&') for idx in sent])[::-1]
        else:
            return " ".join([id2w.get(idx, u'&') for idx in sent])
    
    def cal(self, n_example=5):
        train_n_example = int(n_example / 2)
        test_n_example = n_example - train_n_example
        train_examples = random.sample(list(self.dp.X_train), train_n_example)
        test_examples = random.sample(list(self.dp.X_test), test_n_example)
        for _ in range(train_n_example):
            example = self.show(train_examples[_][:-1], self.dp.X_id2w)
            if len(example) < 3:
                continue
            length = random.randint(1, len(example)-2)
            if self.model.reverse:
                o = self.model.infer(example[-length:])[0]
                print('Train_Input: %s | Output: %s | GroundTruth: %s' % (example[-length:], o, example))
            else:
                o = self.model.infer(example[:length])[0]
                print('Train_Input: %s | Output: %s | GroundTruth: %s' % (example[:length], o, example))
       
        for _ in range(test_n_example):
            example = self.show(test_examples[_][:-1], self.dp.X_id2w)
            if len(example) < 3:
                continue
            length = random.randint(1, len(example)-2)
            if self.model.reverse:
                o = self.model.infer(example[-length:])[0]
                print('Train_Input: %s | Output: %s | GroundTruth: %s' % (example[-length:], o, example))
            else:
                o = self.model.infer(example[:length])[0]
                print('Train_Input: %s | Output: %s | GroundTruth: %s' % (example[:length], o, example))
        print("")
    """    
    def test_bleu(self, N=300, gram=4):
        all_score = []
        for i in range(N):
            input_indices = self.show(self.dp.X_test[i][:-1], self.dp.X_id2w)
            o = self.model.infer(input_indices)[0]
            refer4bleu = [[' '.join([self.dp.X_id2w.get(w, u'&') for w in self.dp.X_test[i]])]]
            candi = [' '.join(w for w in o)]
            score = BLEU(candi, refer4bleu, gram=gram)
            all_score.append(score)
        return np.mean(all_score)
    """
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
        
    