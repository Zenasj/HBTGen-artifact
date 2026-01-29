# tf.random.uniform((B, 100, 100, 3), dtype=tf.float32)  # Assuming batch size B is dynamic and input shape is (100, 100, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instead of using a Lambda layer wrapping tf.image.per_image_standardization,
        # we will directly call tf.image.per_image_standardization in call() method.
        # This avoids serialization issues with Lambda + this TF op that occur in TF 2.3.
        # We add two Dense layers as in the original example.

        # Define Dense layers
        self.dense1 = tf.keras.layers.Dense(100, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        # Apply per-image standardization directly, not via Lambda
        # This aligns with the recommended fix from the issue.
        x = tf.image.per_image_standardization(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (B, 100, 100, 3)
    # Assuming batch size 4 for example; can be adjusted or made dynamic.
    return tf.random.uniform((4, 100, 100, 3), dtype=tf.float32)

