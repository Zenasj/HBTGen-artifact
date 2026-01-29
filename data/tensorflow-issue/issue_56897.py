# tf.random.uniform((B, 11), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the same model architecture as the original PredictedDestination.model
        self.dense_0 = tf.keras.layers.Dense(32, name='input')
        self.dense_1 = tf.keras.layers.Dense(16, activation=tf.nn.relu, name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(8, activation=tf.nn.relu, name='dense_2')
        self.dense_3 = tf.keras.layers.Dense(4, activation=tf.nn.relu, name='dense_3')
        self.dense_out = tf.keras.layers.Dense(2, name='output')

    def call(self, x):
        x = self.dense_0(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        logits = self.dense_out(x)
        probabilities = tf.nn.softmax(logits, axis=-1)
        # Return both probabilities and logits as dictionary, matching the saved signature
        return {
            "output": probabilities,
            "logits": logits
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input with shape [batch_size, 11]
    # Batch size is chosen as 4 arbitrarily, matching None dimension of input signature
    return tf.random.uniform((4, 11), dtype=tf.float32)

