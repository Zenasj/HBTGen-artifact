# tf.random.uniform((20, 7, 7, 1280), dtype=tf.float32) ← Inferred input shape matches base MobileNetV2 output to head

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base model simulating MobileNetV2 feature extractor output shape (7,7,1280)
        # This is a placeholder for bases.MobileNetV2Base(image_size=224)
        # In practice this base would be the frozen feature extractor.
        # Here we simulate the output of such a base to fit the described head input shape.
        self.base_output_shape = (7, 7, 1280)

        # Define the head according to the issue example
        self.head = tf.keras.Sequential([
            layers.Flatten(input_shape=self.base_output_shape),  # flatten base output
            layers.Dense(
                units=32,
                activation='relu',
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01)),
            layers.Dense(
                units=4,
                activation='softmax',
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01)),
        ])

    def call(self, inputs, training=False):
        # inputs expected to be from the base's output: shape (batch, 7,7,1280)
        # forward pass through head
        return self.head(inputs, training=training)


def my_model_function():
    # Return an instance of MyModel, weights uninitialized as in the example
    return MyModel()

def GetInput():
    # Return a random tensor input matching the base output: batch size=20 (train_batch_size)
    # Shape from example: (20, 7, 7, 1280)
    return tf.random.uniform((20, 7, 7, 1280), dtype=tf.float32)

