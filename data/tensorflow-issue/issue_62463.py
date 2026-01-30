import tensorflow as tf
from tensorflow import keras

class Model1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Tensor objects (with comments for shapes)
        self.p0 = tf.Variable(params[0]) # [1] float32
        self.p1 = tf.Variable(params[1]) # [] float32

        # Layers or other Keras model objects

    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [52, 33, 4] : float32
        v9_0 = tf.add(self.p1, self.p0)
        v5_0 = tf.multiply(v9_0, inp)
        v0_0 = tf.negative(v5_0)
        v11_0 = tf.add(v5_0, v0_0)
        return v11_0

class Model2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Tensor objects (with comments for shapes)
        self.p0 = tf.Variable(params[0]) # [1] float32
        self.p1 = tf.Variable(params[1]) # [] float32

        # Layers or other Keras model objects

    @tf.function(jit_compile=True)
    def __call__(self, inp):
        # Forward pass logic using TensorFlow operations
        # inp: [52, 33, 4] : float32
        v10_0 = tf.multiply(self.p0, inp)
        v12_0 = tf.multiply(self.p1, inp)
        v13_0 = tf.add(v12_0, v10_0)
        v17_0 = tf.negative(v13_0)
        v38_0 = tf.add(v17_0, v10_0)
        v39_0 = tf.add(v38_0, v12_0)
        return v39_0