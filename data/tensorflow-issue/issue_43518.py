# tf.random.uniform((B, 10), dtype=tf.float32) <- Input shape inferred from "x = np.random.rand(10000, 10)"

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Sequential model layers as given in the issue
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.dense4 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x

def my_model_function():
    # Return an instance of MyModel (weights would normally be loaded externally)
    return MyModel()

def GetInput():
    # Return a batch of inputs with shape (batch_size, 10), here batch_size=32 arbitrarily chosen
    return tf.random.uniform((32, 10), dtype=tf.float32)

