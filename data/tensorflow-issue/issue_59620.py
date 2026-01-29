# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from Flatten input_shape=(28, 28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the MNIST model from the issue
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(256, activation='relu', use_bias=False)
        self.dropout1 = tf.keras.layers.Dropout(0.45)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu', use_bias=False)
        self.dropout2 = tf.keras.layers.Dropout(0.45)
        self.dense3 = tf.keras.layers.Dense(10, use_bias=False)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.dense3(x)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching (batch_size, 28, 28), here batch_size=32 chosen arbitrarily
    # Using dtype tf.float32 to mimic image-like float input
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

