# tf.random.uniform((B, input_dim), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the given example:
        # Input dimension is 256, two dense layers:
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output

def my_model_function():
    # Create a fresh instance of the same model architecture
    return MyModel()

def GetInput():
    # The sample input shape from the issue is (batch_size, 256)
    # We'll randomly generate a batch of size 64 with float32 values
    input_dim = 256
    batch_size = 64
    # Random values in [0,1), matching np.random.random used in the original example
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

