import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

print(tf.version.GIT_VERSION, tf.version.VERSION) ### v2.4.0-49-g85c8b2a817f 2.4.1

'''
simple autoencoder example
'''

class Encoder(tf.keras.layers.Layer):
    def __init__(self, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.enc = tf.keras.layers.Dense(units=20, activation="relu", name='encoder_layer')

    def call(self, inputs):
        x = inputs
        x = self.enc(x)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dec = tf.keras.layers.Dense(units=1000, activation="sigmoid", name='decoder_layer')

    def call(self, inputs):
        x = inputs
        x = self.dec(x)
        return x
    
class AutoEncoder(tf.keras.Model):
    def __init__(self, name="autoencoder", **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_pred), axis=-1), axis=-1) # mean squared error
        gradients = tape.gradient(mse_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": mse_loss}
    
'''
example run
'''
inp = np.random.rand(10000,1000)
ae = AutoEncoder()
ae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3))
ae.fit(inp, inp)

'''
saving throws warning message "Found untraced functions ... These functions will not be directly callable after loading."
This is new in TF 2.4.
TF 2.3 did not have this issue.
'''
ae.save('tmp')