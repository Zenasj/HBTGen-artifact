import random
from tensorflow.keras import optimizers

import os
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras


os.environ['AUTOGRAPH_VERBOSITY'] = '10'

def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class TFConv1D(layers.Layer):
    def __init__(self, input_dim, output_dim, init_std=0.02, use_bias=True, **kwargs):
        """ TFConv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super(TFConv1D, self).__init__(**kwargs)
        self.nf = output_dim
        self.nx = input_dim
        self.initializer_range = init_std
        self.use_bias = use_bias
        self.weight = self.add_weight(
            "{}_weight".format(self.name),
            shape=[self.nx, self.nf],
            initializer=keras.initializers.TruncatedNormal(stddev=init_std))
        if self.use_bias:
            self.bias = self.add_weight(
                "{}_bias".format(self.name),
                shape=[1, self.nf],
                initializer=tf.zeros_initializer())

    def call(self, x):
        x = tf.matmul(x, self.weight)
        if self.use_bias:
            x += self.bias
        return x


class Adaptive_Softmax(layers.Layer):
    def __init__(self, vocab_size: int, hidden_dim: int, cutoffs: list, padding_index: int, init_std=0.02):
        super(Adaptive_Softmax, self).__init__()
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_clusters = len(cutoffs) + 1
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.cluster_logit = TFConv1D(hidden_dim, self.n_clusters)
        self.logits = self.add_weight(
            "{}_weight".format(self.name),
            shape=[hidden_dim, vocab_size],
            initializer=keras.initializers.TruncatedNormal(stddev=init_std))

        self.bias = self.add_weight(
            "{}_bias".format(self.name),
            shape=[1, vocab_size],
            initializer=tf.zeros_initializer())

    def call(self, x, y):
        x = x[:, :-1]
        b, l, h = shape_list(x)
        x = tf.reshape(x, [b * l, -1])
        y = tf.reshape(y, [-1])
        cl = self.cluster_logit(x)
        cluster_ll = tf.nn.log_softmax(cl, axis=1)
        nll = tf.zeros_like(y, dtype=x.dtype)
        tail_weight = self.logits

        for i in range(self.n_clusters):
            l, r = self.cutoffs[i], self.cutoffs[i + 1]
            mask = (y >= l) & (y < r)
            indices = tf.where(mask)
            target_i = tf.boolean_mask(y, mask) - l
            tail_logit = tf.matmul(tf.boolean_mask(x, mask), tail_weight[:, l:r]) + self.bias[:, l:r]
            tail_logprob_i = tf.nn.log_softmax(tail_logit, axis=1)  # [b,vocab]
            # word_nll[indices] = -logprob_i
            cur_ll = tf.gather_nd(cluster_ll, tf.concat([indices, tf.ones_like(indices) * i], 1)) + \
                     tf.gather_nd(tail_logprob_i,
                                  tf.stack([tf.range(tf.size(target_i, out_type=target_i.dtype)), target_i], 1))
            nll = tf.tensor_scatter_nd_update(nll, indices, -cur_ll)
        return nll

vocab_size = 51
hidden_dim = 100
cutoffs = [5,20]
padding_index = 50
x = tf.random.normal((800,51,100),dtype=tf.float32)
y = tf.random.uniform((800,50),maxval=50,dtype=tf.int64)

dataset = tf.data.Dataset.from_tensor_slices((x,y))
batchfier = dataset.batch(4)

model = Adaptive_Softmax(vocab_size,hidden_dim,cutoffs,padding_index)
optimizer = keras.optimizers.Adam()

@tf.function
def update_step(x, y):
    with tf.GradientTape() as tape:
        batch_loss = model(x,y)
    step_grad = tape.gradient(batch_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(step_grad, model.trainable_variables))
    return batch_loss

for x,y in batchfier:
    update_step(x,y)

cur_ll = tf.gather_nd(cluster_ll, tf.concat([indices, tf.ones_like(indices) * i], 1)) + \
                     tf.gather_nd(tail_logprob_i,
                                  tf.stack([tf.range(tf.size(target_i, out_type=target_i.dtype)), target_i], 1))

cur_ll = (
                tf.gather_nd(cluster_ll, tf.concat([indices, tf.ones_like(indices) * i], 1)) +
                tf.gather_nd(tail_logprob_i,
                             tf.stack([tf.range(tf.size(target_i, out_type=target_i.dtype)), target_i], 1)))