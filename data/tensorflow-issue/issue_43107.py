import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

class DenseCov(tf.keras.layers.Dense):
    def __init__(self, units, segments, **kwargs):
        
        self.segment_ids = tf.keras.backend.constant(segments, dtype=tf.int64, name='segment_ids')
        self.W_ragged = None
        self.embed_dim = None
        super().__init__(units=units, use_bias=False, kernel_initializer=tf.keras.initializers.Ones(), **kwargs)
    
    def call(self, inputs):
        logits = super().call(inputs)

        # Want to minimize Kullback–Leibler divergence between two multivariate normal distributions
        # But for simplicity minimize only trace

        W_ragged = tf.RaggedTensor.from_value_rowids(tf.transpose(self.kernel), self.segment_ids, name='init_ragged')
        means = tf.reduce_mean(W_ragged, axis=1, name='means')
        W_centred = W_ragged - tf.expand_dims(means, 1)
        cov_matrix = tf.map_fn(lambda x: tf.matmul(x, x, True) / (tf.cast(tf.shape(x)[0], tf.float32) - 1), W_centred, name='calc_covar')
        cov_matrix = cov_matrix.to_tensor(name='convert_dense')
        
        traces =  tf.map_fn(lambda x: tf.linalg.trace(x), cov_matrix, name='calc_trace')
        # traces = 0.

        # logdets = tf.map_fn(lambda x: -tf.linalg.logdet(x), cov_matrix, name='calc_logdet')  # disabled for simplicity
        logdets = 0.

        # loss = traces + logdets + tf.reduce_sum(centroid_means**2, axis=1)  # disabled for simplicity
        loss = traces

        loss = tf.reduce_mean(loss, name='total_loss')        
        self.add_loss(loss)
        return logits
    
    def build(self, input_shape):
        super().build(input_shape)

        # Distorted matrix a bit
        W = self.kernel
        W.assign_add(np.random.randn(N_dim, N_class).astype(np.float32))
        
    def get_config(self):
        config = {
            'segments': self.segment_ids.numpy(),
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config

N_class, N_domains, N_dim, N_samples = 50000, 1, 10, 100
segments = np.random.randint(0, N_domains, N_class)
segments = np.sort(segments)

data = np.random.randn(N_samples, N_dim).astype(np.float32)
y_true = np.random.randn(N_samples).astype(np.float32)

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(N_dim,)))
model.add(DenseCov(N_class, segments))
model.add(tf.keras.layers.Dense(1))

def dummy_loss(y_true, y_pred):
    return 0*tf.reduce_sum(y_pred)

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss=dummy_loss)

from tensorflow.raw_ops import RaggedTensorToVariant

@tf.RegisterGradient("RaggedTensorFromVariant")
def _RaggedTensorFromVariantGrad(*args):
    if len(args) == 2:
        op, grad = args
        res = [RaggedTensorToVariant(rt_nested_splits=[], rt_dense_values=grad,
                                      batched_input=False)]
    else:
        op, empty, grad = args
        res = [RaggedTensorToVariant(rt_nested_splits=[op.outputs[0]], rt_dense_values=grad,
                                    batched_input=True)]
    return res

initial_trace = np.trace(np.cov((model.layers[-2].weights[0].numpy())))

loss = []
for i in range(10):
    res = model.train_on_batch(data[:10], y_true[:10])
    print(f"Iter {i}, loss: {res}, delta: {initial_trace-res} ")

import tensorflow as tf
import numpy as np

class DenseCov(tf.keras.layers.Dense):
    def __init__(self, units, segments, **kwargs):
        
        self.segment_ids = tf.keras.backend.constant(segments, dtype=tf.int64, name='segment_ids')
        self.W_ragged = None
        self.embed_dim = None
        super().__init__(units=units, use_bias=False, kernel_initializer=tf.keras.initializers.Ones(), **kwargs)
    
    def call(self, inputs):
        logits = super().call(inputs)

        # Want to minimize Kullback–Leibler divergence between two multivariate normal distributions
        # But for simplicity minimize only trace

#         W_ragged = tf.RaggedTensor.from_value_rowids(tf.transpose(self.kernel), self.segment_ids, name='init_ragged')
        W_ragged = tf.transpose(self.kernel)
        W_ragged = tf.expand_dims(W_ragged, 0)

        means = tf.reduce_mean(W_ragged, axis=1, name='means')
        W_centred = W_ragged - tf.expand_dims(means, 1)
        cov_matrix = tf.map_fn(lambda x: tf.matmul(x, x, True) / (tf.cast(tf.shape(x)[0], tf.float32) - 1),
                               W_centred, name='calc_covar')
#         cov_matrix = cov_matrix.to_tensor(name='convert_dense')
        
        traces =  tf.map_fn(lambda x: tf.linalg.trace(x), cov_matrix, name='calc_trace')
        # traces = 0.

        # logdets = tf.map_fn(lambda x: -tf.linalg.logdet(x), cov_matrix, name='calc_logdet')  # disabled for simplicity
        logdets = 0.

        # loss = traces + logdets + tf.reduce_sum(centroid_means**2, axis=1)  # disabled for simplicity
        loss = traces

        loss = tf.reduce_mean(loss, name='total_loss')        
        self.add_loss(loss)
        return logits
    
    def build(self, input_shape):
        super().build(input_shape)

        # Distorted matrix a bit
        W = self.kernel
        W.assign_add(np.random.randn(N_dim, N_class).astype(np.float32))
        
    def get_config(self):
        config = {
            'segments': self.segment_ids.numpy(),
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config

N_class, N_domains, N_dim, N_samples = 500, 1, 10, 100
segments = np.random.randint(0, N_domains, N_class)
segments = np.sort(segments)

data = np.random.randn(N_samples, N_dim).astype(np.float32)
y_true = np.random.randn(N_samples).astype(np.float32)

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(N_dim,)))
model.add(DenseCov(N_class, segments))
model.add(tf.keras.layers.Dense(1))

def dummy_loss(y_true, y_pred):
    return 0*tf.reduce_sum(y_pred)

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss=dummy_loss)


initial_trace = np.trace(np.cov((model.layers[-2].weights[0].numpy())))

loss = []
for i in range(10):
    res = model.train_on_batch(data[:10], y_true[:10])
    print(f"Iter {i}, loss: {res}, delta: {initial_trace-res} ")

from tensorflow.raw_ops import RaggedTensorToVariant
@tf.RegisterGradient("RaggedTensorFromVariant")
def _RaggedTensorFromVariantGrad(*args):

    op, empty, grad = args
    res = RaggedTensorToVariant(rt_nested_splits=[op.outputs[0]], rt_dense_values=grad,
                                batched_input=True)
    return res

x = tf.constant([[1., 2.], 
               [3., 4.],
               [-1., 2.], 
               [3., -4.]
            ], dtype=tf.float32)

with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    W_ragged = tf.RaggedTensor.from_value_rowids(x, [0, 0 , 1, 1], name='init_ragged')
    y = tf.reduce_mean(W_ragged, axis=1)
dy_dx = g.gradient(y, W_ragged.values)
print("Gradient means")
print(dy_dx)


with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    W_ragged = tf.RaggedTensor.from_value_rowids(x, [0, 0 , 1, 1], name='init_ragged')
    y = tf.map_fn(lambda x: tf.matmul(x, x, True) / (tf.cast(tf.shape(x)[0], tf.float32) - 1),
                               W_ragged, name='calc_covar')

dy_dx = g.gradient(y.values, W_ragged.values)
print("\nGradient Covar")
print(dy_dx)