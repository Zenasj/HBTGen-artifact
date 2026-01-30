import random

import os
import collections
import numpy as np
import tensorflow as tf
import pickle

in_dir = "./exp-14"

cur_pth = os.getcwd()
# root_pth = '/'.join(cur_pth.split('/')[:-1])
# out_dir = os.path.join(root_pth, 'exp-14')
out_dir = "./exp-14"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print(out_dir)
else:
    print("The directionary exists")

poem_file = os.path.join(in_dir, 'poem.txt')
print(in_dir, out_dir)

poems = []
with open(poem_file, "r", encoding='UTF-8') as f:
    for line in f:
        try:
            # line = line.decode('UTF-8')
            line = line.strip(u'\n')
            title, content = line.strip(u' ').split(u':')
            content = content.replace(u' ', u'')
            if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = u'[' + content + u']'
            poems.append(content)
        except Exception as e:
            pass

poems = sorted(poems, key=lambda line: len(line))
print('total poems: ', len(poems))
print(poems[:100])

all_words = []
for poem in poems:
    all_words += [word for word in poem]
counter = collections.Counter(all_words)

count_pairs = sorted(counter.items(), key=lambda x: -x[1])

words, _ = zip(*count_pairs)
words = words[:len(words)] + (' ',)
print(words[:20])

word_num_map = dict(zip(words, range(len(words))))
to_num = lambda word: word_num_map.get(word, len(words))
poems_vector = [list(map(to_num, poem)) for poem in poems]
print(poems[:10])
print(poems_vector[:10])

w2v_file = os.path.join(out_dir, 'w2v.data')
with open(w2v_file, "wb") as fw:
    pickle.dump([words, word_num_map, poems_vector], fw)

batch_size = 64
n_chunk = len(poems_vector) // batch_size
print(n_chunk)


class DataSet(object):
    def __init__(self, data_size):
        self._data_size = data_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(data_size)

    def next_batch(self, batch_size):

        start = self._index_in_epoch

        if start + batch_size > self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed = self._epochs_completed + 1
            self._index_in_epoch = batch_size
            full_batch_features, full_batch_labels = self.data_batch(0, batch_size)
            return full_batch_features, full_batch_labels

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            full_batch_features, full_batch_labels = self.data_batch(start, end)

            return full_batch_features, full_batch_labels

    def data_batch(self, start, end):
        batches = []
        for i in range(start, end):
            batches.append(poems_vector[self._data_index[i]])

        length = max(map(len, poems_vector))
        xdata = np.full((end - start, length), word_num_map[' '], np.int32)

        for row in range(end - start):
            xdata[row, :len(batches[row])] = batches[row]

        ydata = np.copy(xdata)
        ydata[:, :-1] = xdata[:, 1:]
        return xdata, ydata


def neural_network(batch_size=32, model='lstm', rnn_size=128, num_layers=2, is_training=True):
    tf.reset_default_graph()
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    if is_training:
        output_targets = tf.placeholder(tf.int32, [batch_size, None])
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell

    basicCell = cell_fun(rnn_size, state_is_tuple=True)
    droped_cell = tf.contrib.rnn.DropoutWrapper(basicCell, output_keep_prob=0.5)
    cells = []
    for _ in range(num_layers):
        cells.append(droped_cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    # cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):

        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)])
        softmax_b = tf.get_variable("softmax_b", [len(words)])

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [len(words), rnn_size])

            inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')

    output = tf.reshape(outputs, [-1, rnn_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)

    if is_training:
        return logits, last_state, probs, cell, initial_state, input_data, output_targets
    else:
        return logits, last_state, probs, cell, initial_state, input_data


def load_model(sess, saver, ckpt_path):
    latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
    if latest_ckpt:
        print('resume from', latest_ckpt)
        saver.restore(sess, latest_ckpt)
        return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
    else:
        print('building model from scratch')
        sess.run(tf.global_variables_initializer())
        return -1


model_read_dir = os.path.join(in_dir, 'model')
model_read_path = os.path.join(model_read_dir, 'poem.module')
model_save_dir = os.path.join(out_dir, 'model')
model_save_path = os.path.join(model_save_dir, 'poem.module')


def train_neural_network():
    logits, last_state, _, _, _, input_data, output_targets = neural_network(batch_size)

    targets = tf.reshape(output_targets, [-1])

    with tf.name_scope('sequence_loss'):
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets],
                                                                  [tf.ones_like(targets, dtype=tf.float32)], len(words))
        cost = tf.reduce_mean(loss)
        tf.summary.scalar("loss_value", cost)

    learning_rate = tf.Variable(0.0, trainable=False)

    tvars = tf.trainable_variables()

    #     tf.clip_by_global_norm(
    #     t_list,
    #     clip_norm,
    #     use_norm=None,
    #     name=None
    #     )
    # t_list[i] * clip_norm / max(global_norm, clip_norm)
    # global_norm = sqrt(sum([t**2 for t in t_list]))

    grad_vars = tf.gradients(cost, tvars)
    grads, gn = tf.clip_by_global_norm(grad_vars, 1, name='gradients_clip')

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-4)

    # train_op = optimizer.minimize(cost)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    Session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    Session_config.gpu_options.allow_growth = True

    trainds = DataSet(len(poems_vector))

    summary_merge_op = tf.summary.merge_all()
    with tf.Session(config=Session_config) as sess:
        with tf.device('/gpu:0'):
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.all_variables())

            # last_epoch = load_model(sess, saver, model_read_dir)
            last_epoch = 0
            writer = tf.summary.FileWriter('logs/', sess.graph)
            for epoch in range(last_epoch + 1, 30):

                sess.run(tf.assign(learning_rate, 0.002 * (0.3 ** epoch)))
                # sess.run(tf.assign(learning_rate, 0.01))

                all_loss = 0.0

                for batche in range(n_chunk):

                    x, y = trainds.next_batch(batch_size)

                    # sess.run(train_op, feed_dict={input_data: x, output_targets: y})
                    train_loss, _, _, summary_info = sess.run([cost, last_state, train_op, summary_merge_op],
                                                              feed_dict={input_data: x, output_targets: y})

                    writer.add_summary(summary_info, epoch)

                    grad = sess.run(grads, feed_dict={input_data: x, output_targets: y})
                    all_loss = all_loss + train_loss
                    # print(grad)
                    # print('epoch:', epoch, 'batch:', batche, 'lr:', 0.002 * (0.97 ** epoch), 'tl:', train_loss)
                    if batche % 50 == 1:
                        #     #print(epoch, batche, 0.01,train_loss)
                        print('epoch:', epoch, 'batch:', batche, 'lr:', 0.002 * (0.3 ** epoch), 'tl:', train_loss)

                saver.save(sess, model_save_path, global_step=epoch)

                print('epoch:', epoch, ' Loss: ', all_loss * 1.0 / n_chunk)


train_neural_network()

batch_size=64
n_chunk = len(poems_vector) // batch_size
print(n_chunk)