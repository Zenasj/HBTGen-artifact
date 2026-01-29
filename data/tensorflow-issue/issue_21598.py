# tf.random.uniform((1, 5, 5, 1), dtype=tf.float32) ← input shape from the reported example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model with dilation_rate=2 (expected to observe effect if dilation worked)
        self.convT_dil_1 = tf.keras.layers.Conv2DTranspose(
            filters=8, kernel_size=(2, 2), dilation_rate=2, padding='valid')
        self.convT_dil_2 = tf.keras.layers.Conv2DTranspose(
            filters=8, kernel_size=(3, 3), dilation_rate=2, padding='valid')
        self.convT_dil_3 = tf.keras.layers.Conv2DTranspose(
            filters=8, kernel_size=(3, 3), dilation_rate=2, padding='valid')

        # Model with dilation_rate=1 (no dilation, baseline)
        self.convT_no_dil_1 = tf.keras.layers.Conv2DTranspose(
            filters=8, kernel_size=(2, 2), dilation_rate=1, padding='valid')
        self.convT_no_dil_2 = tf.keras.layers.Conv2DTranspose(
            filters=8, kernel_size=(3, 3), dilation_rate=1, padding='valid')
        self.convT_no_dil_3 = tf.keras.layers.Conv2DTranspose(
            filters=8, kernel_size=(3, 3), dilation_rate=1, padding='valid')

    def call(self, inputs):
        # Forward pass through "dilated" model
        x_dil = self.convT_dil_1(inputs)
        x_dil = self.convT_dil_2(x_dil)
        x_dil = self.convT_dil_3(x_dil)

        # Forward pass through "no dilation" model
        x_no_dil = self.convT_no_dil_1(inputs)
        x_no_dil = self.convT_no_dil_2(x_no_dil)
        x_no_dil = self.convT_no_dil_3(x_no_dil)

        # Compare the two outputs to assess if dilation had any effect
        # The issue reports that despite dilation_rate=2, results equal dilation_rate=1
        # Output the boolean tensor indicating elementwise equality
        are_equal = tf.math.reduce_all(tf.math.equal(x_dil, x_no_dil))
        # Also output the difference norm to quantify difference
        diff_norm = tf.norm(x_dil - x_no_dil)

        # Return a dict-like output with the comparison results
        return {
            'output_with_dilation': x_dil,
            'output_without_dilation': x_no_dil,
            'are_equal': are_equal,
            'difference_norm': diff_norm,
        }

def my_model_function():
    return MyModel()

def GetInput():
    # Mimic the example input shape: batch=1, height=5, width=5, channels=1
    return tf.random.uniform((1, 5, 5, 1), dtype=tf.float32)

