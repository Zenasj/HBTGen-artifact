from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
keras = tf.keras
KL = keras.layers
K = keras.backend

bgr_shape = (128, 128, 3)
#batch_size = 132 #max -tf.1.11.0-cuda 9
batch_size = 86 #max -tf.1.13.1-cuda 10
 
class PixelShuffler(keras.layers.Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):

        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))


        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1
        rh, rw = self.size
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)

        out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
        out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
        out = K.reshape(out, (batch_size, oh, ow, oc))
        return out

    def compute_output_shape(self, input_shape):

        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))


        height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
        width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
        channels = input_shape[3] // self.size[0] // self.size[1]

        if channels * self.size[0] * self.size[1] != input_shape[3]:
            raise ValueError('channels of input and size are incompatible')

        return (input_shape[0],
                height,
                width,
                channels)

    def get_config(self):
        config = {'size': self.size}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
        
def upscale (dim):
    def func(x):
        return PixelShuffler()((KL.Conv2D(dim * 4, kernel_size=3, strides=1, padding='same')(x)))
    return func 
            
inp = KL.Input(bgr_shape)
x = inp
x = KL.Conv2D(128, 5, strides=2, padding='same')(x)
x = KL.Conv2D(256, 5, strides=2, padding='same')(x)
x = KL.Conv2D(512, 5, strides=2, padding='same')(x)
x = KL.Conv2D(1024, 5, strides=2, padding='same')(x)
x = KL.Dense(1024)(KL.Flatten()(x))
x = KL.Dense(8 * 8 * 1024)(x)
x = KL.Reshape((8, 8, 1024))(x)
x = upscale(512)(x)
x = upscale(256)(x)
x = upscale(128)(x)
x = upscale(64)(x)
x = KL.Conv2D(3, 5, strides=1, padding='same')(x)

model = keras.models.Model ([inp], [x])
model.compile(optimizer=keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss='mae')

training_data = np.zeros ( (batch_size,128,128,3) )
loss = model.train_on_batch( [training_data], [training_data] )
print ("FINE")