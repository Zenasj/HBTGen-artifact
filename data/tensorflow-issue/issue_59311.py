import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TestModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._name = "TestModel"
        
        self.kernel_initializer = tf.keras.initializers.HeNormal()
        self.kernel_regularizer = tf.keras.regularizers.l2
        self.l2_reg = 1e-4
        
        self.b_0_0_conv = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1),
                                 strides=(2, 2),
                                 padding="same",
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer(self.l2_reg),
                                 name="in_conv")

        self.b_0_0_bn = tf.keras.layers.BatchNormalization(axis=-1, name="in_bn")
        self.b_0_0_relu = tf.keras.layers.ReLU(name="in_relu")
        self.b_0_0_mp = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="in_mp")

    def call(self, x, mask=None):

        x = self.b_0_0_conv(x)
        x = self.b_0_0_bn(x, training=False)
        x = self.b_0_0_relu(x)
        x = self.b_0_0_mp(x)
        org = x
        return [org, x]