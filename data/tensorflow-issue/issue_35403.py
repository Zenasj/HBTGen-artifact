# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← input shape inferred from issue code input (224,224,3)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct three sequential blocks as described in the issue with given layer names
        self.seq_0 = tf.keras.Sequential(name='seq_0')
        self.seq_0.add(layers.Conv2D(64, (3, 3), strides=2, padding='same', use_bias=False))
        self.seq_0.add(layers.BatchNormalization(name='bn_0'))
        self.seq_0.add(layers.GaussianNoise(stddev=1, name='noise_0'))
        self.seq_0.add(layers.ReLU())

        self.seq_1 = tf.keras.Sequential(name='seq_1')
        self.seq_1.add(layers.Conv2D(64, (3, 3), strides=2, padding='same', use_bias=False))
        self.seq_1.add(layers.BatchNormalization(name='bn_1'))
        self.seq_1.add(layers.GaussianNoise(stddev=1, name='noise_1'))
        self.seq_1.add(layers.ReLU())

        self.seq_2 = tf.keras.Sequential(name='seq_2')
        self.seq_2.add(layers.Conv2D(64, (3, 3), strides=2, padding='same', use_bias=False))
        self.seq_2.add(layers.BatchNormalization(name='bn_2'))
        self.seq_2.add(layers.GaussianNoise(stddev=1, name='noise_2'))
        self.seq_2.add(layers.ReLU())

        # Subtract layer to compute output difference between noise and batch norm layers in the first seq block
        self.subtract = layers.Subtract()

    def call(self, inputs, training=False):
        # Forward prop through seq_0, capturing intermediate outputs bn_0 and noise_0
        x = inputs
        # Because we can't easily grab interim layers directly in subclassing, 
        # we manually forward through layers of seq_0 to get bn_0 and noise_0 outputs

        # seq_0 layers: Conv2D, BatchNorm, GaussianNoise, ReLU
        x_conv0 = self.seq_0.layers[0](x)
        bn_0 = self.seq_0.layers[1](x_conv0, training=training)
        noise_0 = self.seq_0.layers[2](bn_0, training=training)
        relu_0 = self.seq_0.layers[3](noise_0)

        # Forward pass seq_1 and seq_2 normally (training param passed to BN and GaussianNoise)
        x = self.seq_1(relu_0, training=training)
        x = self.seq_2(x, training=training)

        # Compute subtraction between noise_0 output and bn_0 output (both tensors)
        sub_0 = self.subtract([noise_0, bn_0])

        # Return a tuple of the final output and the subtraction difference to reflect issue behavior
        return (x, sub_0)

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Return a sample input tensor matching the input shape (batch=1, 224x224 RGB image)
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

