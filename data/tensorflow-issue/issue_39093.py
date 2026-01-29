# tf.random.uniform((B, 3), dtype=tf.float32) ← Input shape inferred from keras.Input((3,)) in the issue

import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        # Custom behavior: add loss only during training.
        # Note: in TF1.x, the `training` parameter may not be properly passed by model.fit.
        # In TF2.x or with run_eagerly=True, this branch is triggered as expected.
        if training:
            self.add_loss(tf.reduce_sum(inputs))
        return inputs


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use the custom layer that conditionally adds loss when training
        self.mylayer = MyLayer()

    def call(self, inputs, training=None):
        # Forward input through custom layer with training param passed
        # This enables conditional loss addition during training vs inference
        x = self.mylayer(inputs, training=training)
        return x


def my_model_function():
    # Return an instance of MyModel; no weights to load
    return MyModel()


def GetInput():
    # Return a random tensor input matching Model's expected input shape: (Batch, 3)
    # Batch size chosen as 2 to match original test
    return tf.random.uniform((2, 3), dtype=tf.float32)

