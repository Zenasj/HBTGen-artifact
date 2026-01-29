# tf.random.uniform((B, 10), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import Dense, Add, LayerNormalization

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # According to issue details, input shape is (None, 10)
        # and model uses three dense layers, then adds inputs + last dense output,
        # then LayerNormalization with modified epsilon to avoid GPU error.
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.dense3 = Dense(10, activation='relu')
        self.add_layer = Add()
        # LayerNormalization epsilon changed to 1e-7 from default 1e-3 to fix GPU compatibility error
        self.norm_layer = LayerNormalization(epsilon=1e-7)
        self.output_layer = Dense(10)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        # Add skip connection with inputs
        x = self.add_layer([x, inputs])
        # Apply LayerNormalization with fixed epsilon
        x = self.norm_layer(x)
        out = self.output_layer(x)
        return out

def my_model_function():
    # Return an instance of MyModel, uninitialized weights
    return MyModel()

def GetInput():
    # Generates a random input tensor matching (batch_size, 10)
    # Batch size arbitrary; for safety choose 8 here
    return tf.random.uniform((8, 10), dtype=tf.float32)

