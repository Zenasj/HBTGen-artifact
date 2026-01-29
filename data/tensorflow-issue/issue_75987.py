# tf.random.uniform((B, 76), dtype=tf.float32) ← inferred input shape based on input_shape=[76] from the original keras.Sequential example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the architecture from the original Sequential model:
        # layers.Dense(128, activation='relu', input_shape=[76]),
        # layers.Dense(64, activation='relu'),
        # layers.Dropout(rate=0.3),
        # layers.Dense(15, activation='softmax'),
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.dense3 = tf.keras.layers.Dense(15, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return self.dense3(x)

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Return a random float32 tensor of shape [batch_size, 76].
    # Using batch size of 32 as a standard reasonable guess.
    return tf.random.uniform((32, 76), dtype=tf.float32)

