# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape inferred from (None, 784) input layer of the model

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(2, activation="relu", name="layer1")

        # The user Lambda function: square_it = lambda x: x ** 2
        def square_it(x):
            return x ** 2

        # output_shape provided by user: (None, None, 10, 20)
        # This is ignored by TensorFlow's internal shape inference, but Keras uses it to help shape inference in eager mode.
        # It does NOT override the output tensor shape.
        def output_of_lambda(input_shape):
            # original attempted custom hardcoded output shape (including unknown batch dim)
            return (None, None, 10, 20)

        # Use the Lambda layer with the output_shape argument,
        # but aware that it won't override TF shape inference
        self.lambda_layer = layers.Lambda(square_it, output_shape=output_of_lambda)

        self.dense2 = layers.Dense(4, name="layer5")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.lambda_layer(x)
        # After Lambda layer, shape remains (batch_size, 2) because tf.shape inference is not overridden
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # According to the original model, input shape is (batch_size, 784)
    # We'll generate a random tensor with a batch size of 8 (arbitrary example)
    return tf.random.uniform((8, 784), dtype=tf.float32)

