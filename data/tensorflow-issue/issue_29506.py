# tf.random.uniform((6720, 700, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model replicates the toy example discussed in the issue:
    - Input shape: (batch_size, timesteps=700, features=3)
    - Single LSTM layer with state size 1
    The example highlights differences in performance related to cuDNN usage in TF2 nightlies.
    """
    def __init__(self):
        super().__init__()
        # LSTM layer with units=1
        self.lstm = tf.keras.layers.LSTM(1)

    def call(self, inputs, training=False):
        return self.lstm(inputs, training=training)

def my_model_function():
    """
    Returns an instance of MyModel, ready for use.
    """
    return MyModel()

def GetInput():
    """
    Generates a random input tensor matching the expected input shape:
    batch size arbitrary (e.g. 8), timesteps=700, feature dim=3,
    dtype tf.float32 to match the example.
    """
    batch_size = 8  # smaller batch size for quick testing
    timesteps = 700
    features = 3
    return tf.random.uniform((batch_size, timesteps, features), dtype=tf.float32)

