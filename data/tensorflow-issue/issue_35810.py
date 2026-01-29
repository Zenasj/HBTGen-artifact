# tf.random.normal((B, L, H), dtype=tf.float32) ← Input shape (batch_size, sequence_length, hidden_dim)

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras


def shape_list(x):
    # Returns shape of tensor as list with dynamic dimensions if static unknown
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class TFConv1D(layers.Layer):
    def __init__(self, input_dim, output_dim, init_std=0.02, use_bias=True, **kwargs):
        """
        TFConv1D layer as used by Radford et al. for OpenAI GPT / GPT-2.
        Works like a Linear layer but with weight transpose.
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
            x = x + self.bias
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
        # x shape: [batch, sequence_length, hidden_dim]
        # y shape: [batch, sequence_length-1]
        # Computes adaptive softmax negative log likelihood loss for targets y given hidden representations x
        
        x = x[:, :-1]  # Remove last timestep to align with targets' length
        b, l, h = shape_list(x)
        x_flat = tf.reshape(x, [b * l, -1])  # flatten batch and seq dims for matmul
        y_flat = tf.reshape(y, [-1])  # flatten targets

        # Compute cluster logits (log-prob of cluster membership)
        cl = self.cluster_logit(x_flat)  # shape [b*l, n_clusters]
        cluster_ll = tf.nn.log_softmax(cl, axis=1)  # log probs for clusters

        nll = tf.zeros_like(y_flat, dtype=x.dtype)  # init negative log likelihood vector
        
        tail_weight = self.logits  # weights for tail softmax per vocab token

        # Compute loss for each cluster separately
        for i in range(self.n_clusters):
            lcut, rcut = self.cutoffs[i], self.cutoffs[i + 1]
            mask = (y_flat >= lcut) & (y_flat < rcut)
            indices = tf.where(mask)
            target_i = tf.boolean_mask(y_flat, mask) - lcut
            if tf.size(target_i) == 0:
                continue  # no targets in this cluster, skip
            
            # Compute tail logits for selected examples and their subset of vocab
            selected_x = tf.boolean_mask(x_flat, mask)
            tail_logit = tf.matmul(selected_x, tail_weight[:, lcut:rcut]) + self.bias[:, lcut:rcut]
            tail_logprob_i = tf.nn.log_softmax(tail_logit, axis=1)  # [n_selected, vocab_subset]

            # Compute combined log likelihood for each selected sample:
            # log p(cluster) + log p(target | cluster)
            cur_ll = (
                tf.gather_nd(cluster_ll, tf.concat([indices, tf.ones_like(indices) * i], axis=1)) +
                tf.gather_nd(tail_logprob_i,
                             tf.stack([tf.range(tf.size(target_i, out_type=target_i.dtype)), target_i], axis=1))
            )
            # Update negative log likelihood vector at proper indices
            nll = tf.tensor_scatter_nd_update(nll, indices, -cur_ll)
        return nll


class MyModel(tf.keras.Model):
    # Encapsulates the Adaptive_Softmax model for simplicity
    def __init__(self):
        super(MyModel, self).__init__()
        # Use the same parameters as in the issue example
        self.vocab_size = 51
        self.hidden_dim = 100
        self.cutoffs = [5, 20]
        self.padding_index = 50
        self.adaptive_softmax = Adaptive_Softmax(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            cutoffs=self.cutoffs,
            padding_index=self.padding_index,
            init_std=0.02
        )

    def call(self, inputs):
        # Expect inputs as tuple: (x, y)
        x, y = inputs
        # Return the negative log likelihood loss tensor (shape: [batch*seq])
        return self.adaptive_softmax(x, y)


def my_model_function():
    return MyModel()


def GetInput():
    # Create random input tensors matching the shapes in the original issue:
    # x: [batch=4, sequence_length=51, hidden_dim=100] (float32 normal)
    # y: [batch=4, sequence_length=50] (int64 targets in [0,50))
    x = tf.random.normal((4, 51, 100), dtype=tf.float32)
    y = tf.random.uniform((4, 50), maxval=50, dtype=tf.int64)
    return (x, y)

