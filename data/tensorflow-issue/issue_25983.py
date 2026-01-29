# tf.random.uniform((B, 10), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers as per the issue's example Sequential model
        # Input shape is (10,)
        self.dense1 = tf.keras.layers.Dense(50, activation='relu')
        self.dense2 = tf.keras.layers.Dense(3, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming batch size 4 for example
    return tf.random.uniform((4, 10), dtype=tf.float32)

