# tf.random.uniform((10,), dtype=tf.float32) ← Input shape inferred from example train_dataset shape (10,)

import tensorflow as tf

class ProdLayer(tf.keras.layers.Layer):
    def __init__(self, name):
        super(ProdLayer, self).__init__()
        # Using a trainable variable initialized to 0.01, per original code
        # Note: Using tf.Variable via tf.keras.backend.variable to be consistent with snippet, but tf.Variable preferred.
        self.w = tf.Variable(0.01, trainable=True, name='var_' + name, dtype=tf.float32)

    def call(self, x):
        return x * self.w

class SumLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SumLayer, self).__init__()

    def call(self, x1, x2):
        return x1 + x2

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Place w1 on GPU 0
        with tf.device('/GPU:0'):
            self.L1 = ProdLayer('w1')

        # Place w2 on GPU 1
        with tf.device('/GPU:1'):
            self.L2 = ProdLayer('w2')

        # w3 intended to be mirrored variable on both GPUs
        # Because true mirroring via MirroredStrategy scope with manual device placement is problematic,
        # we create w3 as a variable which is not tied to a single device here.
        # To simulate, place w3 on CPU (or let TF place it)
        # This follows the suggestion that mixed manual device placement with MirroredStrategy variable creation breaks.
        self.w3 = tf.Variable(0.01, trainable=True, name='var_w3', dtype=tf.float32)

        # SumLayer placed on GPU 0 (perform sum there)
        with tf.device('/GPU:0'):
            self.L3 = SumLayer()

    def call(self, input_layer):
        # Compute w1 * x + w3 on GPU 0
        with tf.device('/GPU:0'):
            y1 = self.L1(input_layer) + self.w3

        # Compute w2 * x + w3 on GPU 1
        with tf.device('/GPU:1'):
            y2 = self.L2(input_layer) + self.w3

        # Sum y1 + y2 on GPU 0
        with tf.device('/GPU:0'):
            y_hat = self.L3(y1, y2)

        return y_hat

def my_model_function():
    # Return an instance of MyModel without usage of MirroredStrategy scope
    return MyModel()

def GetInput():
    # From original snippet, input is a vector of floats with shape (10,)
    # The example train_dataset used np.random.random((10,))
    return tf.random.uniform((10,), dtype=tf.float32)

