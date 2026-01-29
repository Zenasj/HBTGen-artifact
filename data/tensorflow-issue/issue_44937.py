# tf.random.uniform((B, WINDOW_SIZE-1, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructed from Sequential architecture described in the issue
        # Input shape: (WINDOW_SIZE-1, 1)
        self.gru1 = tf.keras.layers.GRU(50, return_sequences=True)
        self.gru2 = tf.keras.layers.GRU(50)
        self.dense1 = tf.keras.layers.Dense(10, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(1, activation='tanh')
        # Lambda layer to scale output by 100 (to match output scale)
        self.scale = tf.keras.layers.Lambda(lambda x: x * 100)

    def call(self, inputs):
        x = self.gru1(inputs)
        x = self.gru2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.scale(x)
        return x

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # We infer from the code that input shape is (WINDOW_SIZE-1, 1),
    # batch dimension unknown, but batch dimension must be included.
    # For this, we generate a random batch of examples.
    # Since typical batch size used in training is 32, we use that.
    B = 32
    WINDOW_SIZE = 10  # from the description in the issue
    # Create random input tensor with values in [0,1)
    return tf.random.uniform((B, WINDOW_SIZE - 1, 1), dtype=tf.float32)

