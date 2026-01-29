# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Layers matching the reported functional model in the issue:
        # Input shape: (28, 28, 1)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax', name='label')

    def call(self, inputs, training=False):
        # Forward pass through the layers, respecting training flag for dropout
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random batch input with shape (batch_size, 28, 28, 1)
    # Using batch size 32 as an example. dtype is float32 to match image input
    batch_size = 32
    return tf.random.uniform(shape=(batch_size, 28, 28, 1), dtype=tf.float32)

