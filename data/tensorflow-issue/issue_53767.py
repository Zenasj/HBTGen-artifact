# tf.random.uniform((1, 6, 8, 48), dtype=tf.float32) ← inferred input shape from the issue's example inp = tf.keras.Input(shape=(6,8,48), batch_size=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This is a fused implementation from the issue reproducing the quantized transposed conv model.
        # Following the provided QuantConv2DTransposed layer with per-channel fake quantization on kernel.
        # Added comments and explicit shape inference for clarity.

        # Kernel shape: [3, 3, input_channels=48, output_channels=24]
        self.kernel_shape = [3, 3, 48, 24]

        # Create the trainable kernel weight:
        # Note: The issue layer uses add_weight with shape [3,3,input_channels,24]
        # Keep this as a trainable weight.
        self.kernel = self.add_weight(
            name="kernel",
            shape=self.kernel_shape,
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )

        # Parameters for fake quantization ranges (per-channel)
        self.min_vals = -3.0 * tf.ones([24], dtype=tf.float32)
        self.max_vals = 3.0 * tf.ones([24], dtype=tf.float32)

    def call(self, inputs):
        """
        Forward pass mirrors the original layer call logic:
        1. Apply per-channel fake quantization on kernel weights.
        2. Transpose kernel dims from (3,3,in_c,out_c) to (3,3,out_c,in_c) for conv2d_transpose.
        3. Perform conv2d_transpose with stride=1, output shape=inputs.shape[:-1] + [24].
        4. Inputs are expected to be already quantized by fake_quant_with_min_max_vars.
        """

        # Apply per-channel fake quantization to self.kernel
        # narrow_range=True matches the original code
        quant_filters = tf.quantization.fake_quant_with_min_max_vars_per_channel(
            self.kernel,
            min=self.min_vals,
            max=self.max_vals,
            narrow_range=True,
        )
        # Transpose filters: (3,3,in_c,out_c) -> (3,3,out_c,in_c)
        # Needed for tf.nn.conv2d_transpose which expects [H, W, output_channels, in_channels].
        filters_transposed = tf.transpose(quant_filters, perm=(0, 1, 3, 2))

        # Compute output shape: keep batch and spatial dimensions from inputs
        # inputs.shape = [batch, height, width, channels]
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        out_channels = 24

        output_shape = tf.stack([batch_size, height, width, out_channels])

        # Perform transposed convolution with stride = 1 and SAME padding (default)
        # Note: original code uses stride=1 without padding specified, the default is VALID in tf.nn.conv2d_transpose
        # To exactly reproduce behavior, use stride=1 and padding='SAME' assuming the intent is to preserve spatial shape.
        conv_transpose_output = tf.nn.conv2d_transpose(
            inputs,
            filters_transposed,
            output_shape,
            strides=1,
            padding='SAME',
        )
        return conv_transpose_output


def my_model_function():
    # Returns an instance of MyModel.
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the expected input:
    # Shape: batch=1, height=6, width=8, channels=48 (as per the original example)
    # Using float32 values in the range [-3, 3] to match fake quant range.
    return tf.random.uniform([1, 6, 8, 48], minval=-3.0, maxval=3.0, dtype=tf.float32)

