from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class Inner(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(8,
            (3, 3),
            kernel_initializer=tf.keras.initializers.he_normal(),
            padding='same',
            name='conv1')

    def call(self, x, dummy=False):
        x = self.conv1(x)
        return x

class Outer(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.down = tf.keras.layers.Conv2D(8,
            (3, 3),
            strides=(2, 2),
            kernel_initializer=tf.keras.initializers.he_normal(),
            padding='same',
            name='down')

        self.inner = Inner()

    def call(self, x, dummy=False):
        x_small = self.down(x)
        return self.inner(x, dummy=dummy), self.inner(x_small, dummy=dummy)

class Infer(tf.Module):
    def __init__(self):
        super().__init__()

        # Decorate the inference function with tf.function
        self.infer_ = tf.function(self.infer, input_signature=[
             tf.TensorSpec([1, 64, 64, 8], tf.float32, 'prev_img')])

        self.outer = Outer()

    def infer(self, input):
        return self.outer(input, False)

# Create model
infer = Infer()

# Save the trained model
signature_dict = {'infer': infer.infer_}
saved_model_dir = '/tmp/saved_model'
tf.saved_model.save(infer, saved_model_dir, signature_dict)