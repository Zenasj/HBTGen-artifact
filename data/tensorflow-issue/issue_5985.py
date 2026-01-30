# #!/usr/bin/env python3

# python 3.5 (anaconda)
# TF 0.12 RC from https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp35-cp35m-linux_x86_64.whl
# $ python -c "import tensorflow; print(tensorflow.__version__)"
#  0.12.0-rc0

# OS: RedHat 7.2
# CPU version only, no GPU, no CUDA, no CUDNN

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell


BATCH = 5  # batch size
MAX_LEN = 10  # max length of the sequence
MLP_HIDDEN_DIM = 128  # number of hidden neurons in the MLP
EMBEDDING_DIM = 300  # embedding dimension
VOCAB_SIZE = 8  # vocabulary size

THREADS = 4  # number of threads to be used
STD=0.001  # standard deviation of ariable initializers


class SimpleSentiment:
    def __init__(self, adversarial=False, device='/cpu:0'):

        self.embeddings = tf.get_variable('word_embeddings',
                                          initializer=tf.random_uniform([VOCAB_SIZE, EMBEDDING_DIM], -1.0, 1.0))

        with tf.variable_scope('sentiment') as scope:
            with tf.device(device):
                # Inputs
                self.text = tf.placeholder(tf.int32, [BATCH, MAX_LEN])
                self.text_len = tf.placeholder(tf.int32, [BATCH])
                self.sentiment = tf.placeholder(tf.int32, [BATCH])

                # Normal loss
                loss_normal = self._loss(self.text, self.text_len, self.sentiment)

                # Define the optimizer
                # Note: I've tried multiple of optimizers and none helped
                optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)

                if adversarial:  # Define adversarial loss
                    # Let's acces already defined variables
                    scope.reuse_variables()

                    # Gradients of all variable (according to normal loss)
                    gradients = optimizer.compute_gradients(loss_normal)
                    print(len(gradients), gradients)

                    # gradients of the embeddings
                    emb_gradient = optimizer.compute_gradients(loss_normal, [self.embeddings])[0][0]

                    # this how much we want to shift the embeddings, i.e. going "against" the gradient
                    delta = 0.001*tf.sign(emb_gradient)

                    # let's compute the loss once again but this time we add the delta to the embeddings
                    loss_adversarial = self._loss(self.text, self.text_len, self.sentiment, delta)

                    # new gradient of the whole computational graph
                    adversarial_gradients = optimizer.compute_gradients(loss_adversarial)
                    print(len(adversarial_gradients), adversarial_gradients)  # everything is None!

                    # Now we compute an average of old and new gradients
                    new_gradients = [((g + ag)/2, vg) for ((g, vg), (ag, avg)) in zip(gradients, adversarial_gradients)]

                    # and apply them
                    self.training = optimizer.apply_gradients(new_gradients)

                    # Btw this doesn't work either
                    # self.training = optimizer.apply_gradients(adversarial_gradients)

                    self.loss_final = (loss_normal + loss_adversarial) / 2

                else:  # Normal loss
                    # simply minimize according to the gradients
                    self.loss_final = loss_normal
                    self.training = optimizer.minimize(loss_normal)

                # Create the session
                self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=THREADS,
                                                                intra_op_parallelism_threads=THREADS,
                                                                allow_soft_placement=True))

                # init everything (still deprecated way)
                self.session.run(tf.initialize_all_variables())

    def _loss(self, text, text_len, sentiment, emb_delta=0):
        # use embedding
        # note that emb_delta is zero as long as adversarial=False
        # if adversarial=False then each embedding is shifted by appropriate emb_delta
        text = tf.nn.embedding_lookup(self.embeddings + emb_delta, text)

        # run gru
        gru_cell = GRUCell(MLP_HIDDEN_DIM)
        outputs, state = tf.nn.dynamic_rnn(cell=gru_cell,
                                           inputs=text,
                                           sequence_length=text_len,
                                           dtype=tf.float32)

        # define MLP
        W1 = tf.get_variable(name='MLP_W1',
                             shape=[state.get_shape()[1], MLP_HIDDEN_DIM],
                             initializer=tf.random_normal_initializer(mean=0, stddev=STD))
        W2 = tf.get_variable(name='MLP_W2',
                             shape=[MLP_HIDDEN_DIM, 2],
                             initializer=tf.random_normal_initializer(mean=0, stddev=STD))
        h1 = tf.get_variable(name='MLP_h1',
                             shape=[MLP_HIDDEN_DIM],
                             initializer=tf.random_normal_initializer(mean=0, stddev=STD))
        h2 = tf.get_variable(name='MLP_h2',
                             shape=[2],
                             initializer=tf.random_normal_initializer(mean=0, stddev=STD))

        # apply MLP of the last GRU state
        after_first_layer = tf.nn.relu(tf.matmul(state, W1) + h1)
        logits = tf.matmul(after_first_layer, W2) + h2

        # compute loss via categorial cross entropy
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, sentiment))
        return loss


def main():
    for adversarial in [False, True]:
        print('\n=================================')
        if adversarial:
            print('Using the adversarial loss')
        else:
            print('Using the standard loss')

        net = SimpleSentiment(adversarial=adversarial)

        for epoch in range(5):
            print('Epoch {}'.format(epoch))

            for batch in range(3):
                _, loss_final = net.session.run([net.training, net.loss_final],
                                                {net.text: np.array([[3, 1, 1, 2, 1, 0, 0, 0, 0, 0],
                                                                     [3, 4, 1, 2, 1, 4, 4, 0, 0, 0],
                                                                     [1, 1, 1, 2, 0, 0, 0, 0, 0, 0],
                                                                     [3, 3, 3, 2, 1, 7, 0, 0, 0, 0],
                                                                     [7, 1, 5, 2, 4, 2, 2, 2, 1, 7]], dtype='int32'),
                                                 net.text_len: np.array([5, 7, 4, 6, 10], dtype='int32'),
                                                 net.sentiment: np.array([0, 0, 1, 1, 0], dtype='int32')})

                print('\tBatch {}: {}'.format(batch, loss_final))

if __name__ == '__main__':
    main()

def rnn(cell, inputs, initial_state):
  inputs = tf.unpack(inputs, axis=1)
  outputs = []
  state = initial_state
  for input_ in inputs:
    output, state = cell(input_, state)
    outputs.append(output)
  outputs = tf.pack(outputs, axis=1)
  return outputs, state