import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

3
import tensorflow as tf
import numpy as np

class DenseRagged(tf.keras.layers.Layer):
    def __init__(self, 
        units,
        use_bias=True,
        activation = 'linear',
        **kwargs):
        super(DenseRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
        self.units = units
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
                'kernel',
                shape=[last_dim, self.units],
                trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                    'bias',
                    shape=[self.units,],
                    trainable=True)
        else:
            self.bias = None
        super(DenseRagged, self).build(input_shape)
    def call(self, inputs):
        outputs = tf.ragged.map_flat_values(tf.matmul,inputs, self.kernel)
        if self.use_bias:
            outputs = tf.ragged.map_flat_values(tf.nn.bias_add,outputs, self.bias)
        outputs =  tf.ragged.map_flat_values(self.activation,outputs)
        return outputs    

class PoolingRagged(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PoolingRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(PoolingRagged, self).build(input_shape)
    def call(self, inputs):
        node = inputs
        out = tf.math.reduce_mean(node,axis=1)
        return out


data_A = tf.ragged.constant([[[2.0],[ 2.0]], [[3.0]], [[4.0], [5.0],[ 6.0]]] ,ragged_rank=1) 
data_B = tf.ragged.constant([[[4.0],[ 4.0]], [[6.0]], [[8.0], [10.0],[12.0]]],ragged_rank=1) 
data_y = np.array([3.9,5.8,11])
print(data_A.shape,data_B.shape)

in_A = tf.keras.Input(shape=(None,1),dtype ="float32",ragged=True)
out = DenseRagged(1)(in_A)
out = PoolingRagged()(out)
model = tf.keras.models.Model(inputs=in_A, outputs=out)

optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', 'mean_squared_error'])

model.fit(x=data_A,y=data_y,epochs=200)
print("this works")

in_A2 = tf.keras.Input(shape=(None,1),dtype ="float32",ragged=True)
in_B2 = tf.keras.Input(shape=(None,1),dtype ="float32",ragged=True)
outA2 = DenseRagged(1)(in_A2)
outB2 = DenseRagged(1)(in_B2)
outA2 = PoolingRagged()(outA2)
outB2 = PoolingRagged()(outB2)
out2 = tf.keras.layers.Add()([outA2,outB2])
model2 = tf.keras.models.Model(inputs=[in_A2,in_B2], outputs=out2)


optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model2.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', 'mean_squared_error'])

model2.fit(x=[data_A,data_B],y=data_y,epochs=200)
print("this works,too")

in_A3 = tf.keras.Input(shape=(None,1),dtype ="float32",ragged=True)
in_B3 = tf.keras.Input(shape=(None,1),dtype ="float32",ragged=True)
out3 = DenseRagged(1)(in_A3)
out3 = PoolingRagged()(out3)
model3 = tf.keras.models.Model(inputs=[in_A3,in_B3], outputs=out3)


optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model3.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', 'mean_squared_error'])

model3.fit(x=[data_A,data_B],y=data_y,epochs=200)
print("this does not work")