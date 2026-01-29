# tf.random.uniform((B, 32), dtype=tf.float32) ← inferred input shape is (batch_size, 32) based on provided code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers as per the reported model architecture
        self.dense1 = tf.keras.layers.Dense(60, activation='relu')
        self.dense2 = tf.keras.layers.Dense(30, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        outputs = self.dense3(x)
        return outputs

def my_model_function():
    # Return an instance of MyModel, no weights preloaded
    return MyModel()

def GetInput():
    # Return a random tensor input with shape (batch_size=16, 32)
    # Batch size 16 chosen as it was used in original model.fit calls
    return tf.random.uniform((16, 32), dtype=tf.float32)

