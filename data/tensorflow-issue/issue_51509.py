# tf.random.uniform((B, 10), dtype=tf.float32) ← inferred input shape from the example using (10,) shape tuple

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_shape):
        super(MyModel, self).__init__()
        self.my_model_input_shape = input_shape
        self.dense1 = tf.keras.layers.Dense(5, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

        # To properly create the Keras Model with input and output layers for summary and functional support:
        input_layer = tf.keras.layers.Input(self.my_model_input_shape)
        output_layer = self.call(input_layer)
        # Call the parent tf.keras.Model constructor with inputs and outputs to get connected model graph info
        super(MyModel, self).__init__(
            inputs=input_layer,
            outputs=output_layer
        )

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        return self.dense2(x) + x


def my_model_function():
    # Create an instance with the example input shape (10,)
    return MyModel((10,))


def GetInput():
    # Generate a random input tensor matching (batch_size, 10)
    # Batch size is arbitrarily chosen as 1 to keep it simple and compatible
    return tf.random.uniform((1, 10), dtype=tf.float32)

