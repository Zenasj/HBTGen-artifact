# tf.random.uniform((BATCH_SIZE, 784), dtype=tf.float32) ← Input shape inferred from keras.Input(shape=(784,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the model architecture from the issue:
        # 3 dense layers with 4*4096 units each, relu activations, then output dense layer 10 units softmax
        units = 4 * 4096
        self.dense_1 = tf.keras.layers.Dense(units, activation='relu', name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(units, activation='relu', name='dense_2')
        self.dense_3 = tf.keras.layers.Dense(units, activation='relu', name='dense_3')
        self.predictions = tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return self.predictions(x)

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # As per original model input shape (None, 784) with float32 normalized pixel values.
    # We generate a random tensor simulating a batch of BATCH_SIZE=64 examples.
    BATCH_SIZE = 64
    input_shape = (BATCH_SIZE, 784)
    # Use uniform distribution 0-1 to simulate normalized pixel intensities
    return tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)

