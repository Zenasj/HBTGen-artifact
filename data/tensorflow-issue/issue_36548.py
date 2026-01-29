# tf.random.uniform((B, 30), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original Keras model was a Sequential with:
        # Dense(300, relu, input_dim=30) and Dense(4, softmax)
        # We implement that here as layers.
        self.dense1 = tf.keras.layers.Dense(300, activation='relu', input_shape=(30,))
        self.dense2 = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel.
    # Note: weights are randomly initialized; original weights from training are unavailable here.
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel.
    # Original input shape is (batch_size, 30)
    # We pick batch_size=1 to keep it simple.
    # Original input data was scaled using MinMaxScaler to range [0, 255].
    # We'll produce floats in [0, 255] to mimic this scaling.
    return tf.random.uniform(shape=(1, 30), minval=0, maxval=255, dtype=tf.float32)

