# tf.random.uniform((32, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple classification model as per the example in the issue
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Returns an instance of the model
    return MyModel()

def GetInput():
    # Return a random tensor compatible with the model input shape: (32, 28, 28, 1)
    return tf.random.uniform((32, 28, 28, 1), dtype=tf.float32)

