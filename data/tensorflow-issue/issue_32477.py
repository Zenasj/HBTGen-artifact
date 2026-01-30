from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
BatchNormalization._USE_V2_BEHAVIOR = False
import numpy as np

keras = tf.keras


class check_bn_model(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bn = BatchNormalization()

    @tf.function
    def call(self, x, training=None, mask=None):
        x = self.bn(x)
        return x


X = np.ones((10, 5)).astype('float32')
model = check_bn_model()
model.compile('adam', 'mse')
model.fit(X, X, batch_size=2, epochs=2)