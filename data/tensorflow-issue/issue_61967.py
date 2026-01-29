# tf.random.uniform((1, 2), dtype=tf.float32)  ← Input shape inferred from original example: batch size 1, feature dim 2

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__(name="model")
        # Variables initialized exactly as in the original TF Keras model
        self.w1 = tf.Variable([[0.], [0.5]], dtype=tf.float32)
        self.b1 = tf.Variable([-4.], dtype=tf.float32)
        self.r = tf.Variable([-7.], dtype=tf.float32)
        self.c = tf.Variable(1., dtype=tf.float32)  # Not used in call but preserved from original code
        self.m1 = tf.Variable([-4.], dtype=tf.float32)
        self.m2 = tf.Variable([1.], dtype=tf.float32)

    def call(self, x):
        # Replicate original model forward pass exactly:
        # x -> add m1
        x = x + self.m1
        # multiply by r
        x2 = tf.math.multiply(x, self.r)
        # matrix multiply by w1
        x3 = tf.linalg.matmul(x2, self.w1)
        # add b1
        x4 = tf.math.add(x3, self.b1)
        # multiply again by r
        x5 = tf.math.multiply(x4, self.r)
        # add m2
        x6 = tf.math.add(x5, self.m2)
        # multiply again by r
        x7 = tf.math.multiply(x6, self.r)
        # final output
        return x7

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching expected input shape [1, 2] dtype float32
    # Values in range [-10, 10] arbitrary but reasonable for testing
    return tf.random.uniform((1, 2), minval=-10.0, maxval=10.0, dtype=tf.float32)

