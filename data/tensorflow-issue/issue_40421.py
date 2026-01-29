# tf.random.uniform((BATCH, 200, 750), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(200, 750), dropout_rate=0.5):
        super().__init__()
        self.masking = tf.keras.layers.Masking(mask_value=0.0)
        self.gru = tf.keras.layers.GRU(128)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.masking(inputs)
        x = self.gru(x)
        x = self.dropout(x, training=training)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel with default input shape and dropout
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the expected input:
    # Shape: (batch_size=32, timesteps=200, features=750), dtype float32
    # Using batch size 32 as a common default for testing
    return tf.random.uniform((32, 200, 750), dtype=tf.float32)

