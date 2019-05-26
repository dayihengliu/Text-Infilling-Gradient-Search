def _construct_blank(X_indices, num, length, style='random'):
    data = X_indices[:num]
    blank_data = []
    for d in data:
        if style == 'random':
            pos_list = np.sort(random.sample(range(len(d)-2), length)).tolist()
            blank_data.append((d, pos_list))
        elif style == 'middle':
            l = int(length * (len(d) - 1))
            pos_list  = [int((len(d)-1-l) / 2.0) + i for i in range(l)]
            blank_data.append((d, pos_list))
    return blank_data

def show_blank(idx, pos):
    t = [len(id2w)+5 if i in pos else a for i,a in enumerate(idx)]
    s = ' '.join([id2w.get(tt,'_') for tt in t])
    return s

def idx2str(idx, id2w):
    return " ".join(id2w.get(idxx, '_') for idxx in idx)

def str2idx(idx, w2id):
    idx = idx.strip()
    return [w2id[idxx] for idxx in idx.split(' ')]

def cal_candidate_list(model, inputs, pos):
    idx = copy.deepcopy(inputs)
    idx[pos] = model.qid_list[0]
    candidate_list = []
    for t in range(len(model.dp.X_w2id)):
        temp = copy.deepcopy(idx)
        temp = [k if k!=model.qid_list[0] else t for k in temp]
        candidate_list.append(temp)
    return candidate_list


def replace_list(idx, pos_list, target):
    t = [idxx for idxx in idx]
    if target:
        for i,p in enumerate(pos_list):
            t[p] = target[i]
    else:
        for i,p in enumerate(pos_list):
            t[p] = -1
    return t

def cal_optimal(model, idx, pos, max_it=-1):
    X_batch = cal_candidate_list(idx, pos)
    X_batch_len = [len(x) for x in X_batch]
    
    if max_it > 0:
        batch_loss = []
        t = 0
        while t+max_it<len(X_batch):
            batch_loss += model.sess.run(model.batch_loss, {model.X: X_batch[t:t+max_it],
                                                        model.X_seq_len: X_batch_len[t:t+max_it],
                                                        model.output_keep_prob:1,
                                                        model.input_keep_prob:1}).tolist()
            t += max_it
        batch_loss += model.sess.run(model.batch_loss, {model.X: X_batch[t:],
                                                        model.X_seq_len: X_batch_len[t:],
                                                        model.output_keep_prob:1,
                                                        model.input_keep_prob:1}).tolist()
    else:
        batch_loss = model.sess.run(model.batch_loss, {model.X: X_batch,
                                                        model.X_seq_len: X_batch_len,
                                                        model.output_keep_prob:1,
                                                        model.input_keep_prob:1}).tolist()
    #print(len(batch_loss))
    argsort_batch_loss = np.argsort(batch_loss)
    sort_loss = [batch_loss[i] for i in argsort_batch_loss]
    sort_idx = [X_batch[i][pos] for i in argsort_batch_loss]
    return sort_loss, sort_idx

def cal_dist(vector1, vector2):
    return np.linalg.norm(vector1-vector2), np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))


def _init_blank(model, idx, pos):
    c_idx = copy.deepcopy(idx)
    #print(idx, idx2str(idx))
    o_idx = []
    pos = np.sort(pos).tolist()
    for i in range(len(idx)):
        if i in pos:
            if i == 0:
                o_idx.append(w2id[model.generate(1)[1][0][0]])
            else:
                prefix = idx2str(o_idx)
                infer = model.infer(prefix)
                infer = infer[np.argmax([len(inf) for inf in infer])]
                #print('infer',infer,str2idx(infer), i, len(str2idx(infer)))
                if len(str2idx(infer)) <= i:
                    o_idx.append(w2id['<PAD>'])
                else:
                    o_idx.append(str2idx(infer)[i])
        else:
            o_idx.append(c_idx[i])
    init_word = [id2w[o_idx[i]] for i in pos]    
    return o_idx, idx2str(o_idx), init_word

def _init_data(name):
    w2id, id2w = cPickle.load(open('Data/%s/w2id_id2w.pkl' % name,'rb'))
    X_indices = cPickle.load(open('Data/%s/index.pkl' % name,'rb'))
    return X_indices, w2id, id2w

def _init_model(name, lr=5.0, l1_reg_lambda=0.01, l2_reg_lambda=0.01, qid_list=[]):
    rnn_size = dict()
    rnn_size['SM'] = 1024
    rnn_size['Poem'] =  512
    rnn_size['Daily'] = 512 
    rnn_size['APRC'] = 1024
    
    num_layer = dict()
    num_layer['SM'] = 2
    num_layer['Poem'] = 2
    num_layer['Daily'] = 1
    num_layer['APRC'] = 1
    
    max_infer_length = dict()
    max_infer_length['SM'] = 35
    max_infer_length['Poem'] = 33
    max_infer_length['Daily'] = 50
    max_infer_length['APRC'] = 36
    
    model_iter = dict()
    model_iter['SM'] = 30
    model_iter['Poem'] = 30
    model_iter['Daily'] = 30 
    model_iter['APRC'] = 20
    
    assert name in ['SM','Poem','Daily', 'APRC']

    BATCH_SIZE = 256
    NUM_EPOCH = 30
    train_dir ='Model/%s' % name
    dp = LM_DP(X_indices, w2id, BATCH_SIZE, n_epoch=NUM_EPOCH)
    g = tf.Graph() 
    sess = tf.Session(graph=g, config=sess_conf) 
    with sess.as_default():
        with sess.graph.as_default():
            model = LM(
                dp = dp,
                rnn_size = rnn_size[name],
                n_layers = num_layer[name],
                decoder_embedding_dim = rnn_size[name],
                cell_type='lstm',
                max_infer_length = max_infer_length[name],
                att_type='B',
                qid_list = qid_list,
                lr = lr,
                l1_reg_lambda = l1_reg_lambda,
                l2_reg_lambda = l2_reg_lambda,
                is_save = False,
                residual = True,
                is_jieba = False,
                sess=sess
            )
            #print(tf.global_variables())
            #print([var for var in tf.global_variables() if 'Nesterov' in var.name])

    util = LM_util(dp=dp, model=model)
    model.restore('Model/%s/model-%d'% (name,model_iter[name]))
    return model#, tf.global_variables()



def _reload(model, name):
    rnn_size = dict()
    rnn_size['SM'] = 1024
    rnn_size['Poem'] =  512
    rnn_size['Daily'] = 512 
    rnn_size['APRC'] = 1024
    
    num_layer = dict()
    num_layer['SM'] = 2
    num_layer['Poem'] = 2
    num_layer['Daily'] = 1
    num_layer['APRC'] = 1
    
    max_infer_length = dict()
    max_infer_length['SM'] = 35
    max_infer_length['Poem'] = 33
    max_infer_length['Daily'] = 50
    max_infer_length['APRC'] = 36
    
    model_iter = dict()
    model_iter['SM'] = 30
    model_iter['Poem'] = 30
    model_iter['Daily'] = 30 
    model_iter['APRC'] = 20
    
    assert name in ['SM','Poem','Daily', 'APRC']

    model.restore('Model/%s/model-%d'% (name,model_iter[name]))