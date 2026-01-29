# tf.random.uniform((B, 3), dtype=tf.float32) ← Input shape is (batch_size, 3) as inferred from example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers similar to the example
        # Note: input_shape is not necessary here, shape inference happens after first call
        self.fc1 = tf.keras.layers.Dense(4)
        self.fc2 = tf.keras.layers.Dense(3, activation='relu')
        self.fc3 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def my_model_function():
    # Create an instance of the model
    model = MyModel()
    # Since subclass model shape info is undefined before call,
    # we "build" the model by calling it once with input of fixed shape (batch=1)
    dummy_input = tf.zeros((1, 3), dtype=tf.float32)
    model(dummy_input)  # This triggers shape inference and weight creation

    # Now summary can work properly because the model is built
    # (Though it will show output shapes instead of 'multiple')
    return model

def GetInput():
    # Return a random tensor with shape (batch_size, 3)
    # Using batch size 2 here for generality
    return tf.random.uniform((2, 3), dtype=tf.float32)

