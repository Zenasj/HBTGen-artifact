# tf.random.uniform((B, 5, 5, 1), dtype=tf.float32)  # input shape inferred from the example inp in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a custom layer similar to the one in the issue
        self.my_layer = MyLayer()

    def call(self, inputs):
        # Simply forward the call to the custom layer
        return self.my_layer(inputs)

class MyLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        # Fix: build called with correct input shape (not None)
        if input_shape is None:
            print('error: input shape is none')
        else:
            print('build:', input_shape)
        super().build(input_shape)  # Required to set self.built = True

    def call(self, inputs):
        # Print shape for demonstration as in issue
        print('call:', inputs.shape)
        # Simple operation example: multiply inputs by 2
        return inputs * 2

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching (B, 5, 5, 1)
    # B (batch size) is set to 2 as example, dtype float32 to match typical usage
    return tf.random.uniform((2, 5, 5, 1), dtype=tf.float32)

