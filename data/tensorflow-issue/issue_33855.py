# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf
import numpy as np

# This model fuses the example low-level API MNISTModel from the issue discussion,
# which properly handles integer labels (SparseCategoricalCrossentropy),
# casting inputs and outputs correctly.
#
# The model expects input shape [batch, 28, 28, 1] float32 images,
# and outputs logits for 10 classes (float32).
#
# The forward pass returns logits (not argmax), so loss can be applied properly with from_logits=True.
#
# The GetInput function returns a batch of random images suitable for the model input.

class MyModel(tf.keras.Model):
    def __init__(self, n_class=10):
        super(MyModel, self).__init__()
        self.n_class = n_class
        # Convolutional blocks as in example
        self.d0 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.p0 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.d1 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.d2 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.d3 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.d_prior = tf.keras.layers.Dense(64, activation='relu')
        self.d_out = tf.keras.layers.Dense(self.n_class, activation=None)  # logits output, no softmax

    @tf.function
    def call(self, x, training=False):
        # Cast and reshape input to ensure shape [batch,28,28,1], float32
        x = tf.dtypes.cast(x, tf.float32)
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.d0(x)
        x = self.p0(x)
        x = self.d1(x)
        x = self.p1(x)
        x = self.d2(x)
        x = self.p2(x)
        x = self.d3(x)
        x = self.p3(x)
        x = self.flatten(x)
        x = self.d_prior(x)
        logits = self.d_out(x)  # raw logits for all classes (float32)
        return logits

def my_model_function():
    # Instantiate and return the model
    # The standard MNIST setting with 10 classes
    return MyModel(n_class=10)

def GetInput():
    # Return a random batch of [batch, 28, 28, 1] float32 images scaled [0,1]
    BATCH_SIZE = 32  # typical batch size from the examples
    input_tensor = tf.random.uniform((BATCH_SIZE, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)
    return input_tensor

