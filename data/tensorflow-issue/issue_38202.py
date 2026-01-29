# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)  ← Input shape for MNIST grayscale images

import tensorflow as tf

class RadialConstraint(tf.keras.constraints.Constraint):
    """
    Fixed implementation of keras.constraints.RadialConstraint that avoids
    variable creation inside tf.function and replaces K.zeros with K.constant.
    
    This constraint normalizes the kernel weights such that their norm
    within each filter is 1. This implementation is based on the original 
    TF code but fixes the bug causing exceptions during training in TF 2.1+.
    """

    def __call__(self, w):
        """
        Apply the radial constraint on the kernel tensor w.
        This expects a 4D kernel tensor of shape (height, width, input_channels, output_channels).
        """
        # Use Keras backend for operations
        K = tf.keras.backend
        
        kernel_shape = K.int_shape(w)
        # Kernel shape is (H, W, in_channels, out_channels)
        
        # Calculate center index for height/width
        # kernel_shape might contain None, so fallback to w.shape dynamically if needed
        height = kernel_shape[0] if kernel_shape[0] is not None else tf.shape(w)[0]
        width = kernel_shape[1] if kernel_shape[1] is not None else tf.shape(w)[1]

        # For the kernel constraint, original code tries to extract a 3x3 or 2x2 patch
        # depending on even/odd kernel size, applying a normalization by center pixel value.
        # This is the logic fixed here.

        def _kernel_constraint(kernel):
            # kernel shape: (H, W)
            center = height // 2
            start = center

            # Get center pixel value and normalize kernel weights accordingly
            # Handling even/odd kernel sizes differently as original function
            
            # A fixed workaround to replace the problematic K.zeros call:
            zeros_patch = tf.constant([[0, 0], [0, 0]], dtype=kernel.dtype)
            
            # Determine patch2x2 or patch3x3 based on odd/even kernel size (height assumed equal width)
            is_odd = tf.math.floormod(height, 2) == 1

            patch = tf.cond(
                is_odd,
                lambda: kernel[start - 1:start + 2, start - 1:start + 2],  # 3x3 patch for odd size
                lambda: kernel[start - 1:start + 1, start - 1:start + 1] + zeros_patch  # 2x2 patch + zeros for even size
            )
            # Compute norm of patch 
            norm = tf.norm(patch, ord='euclidean')
            norm = tf.where(tf.equal(norm, 0.), tf.constant(1., dtype=norm.dtype), norm)
            normed_kernel = kernel / norm
            return normed_kernel

        # The original implementation applies constraint on each slice along last axis (out_channels)
        # So unstack along last dim, apply constraint fn, then restack

        w_unstack = tf.unstack(w, axis=-1)  # list of (H,W,in_channels) tensors, but we want per filter 2D patch

        # The radial constraint is applied per kernel in kernel stack. 
        # The original code works on each filter kernel (per out_channel), so we need to process accordingly.
        #
        # Note: Kernel shape (H, W, in_channels, out_channels)
        # For each out_channel, we have (H, W, in_channels) kernel.
        # The radial constraint applies on kernel per input channel as well.
        # So actually, original code maps over last axis (out_channels), then inside constraint maps over input channels?

        # To keep it simple and consistent with original code, we replicate original behavior:
        # We apply _kernel_constraint on each kernel slice of shape (H,W), that means per channel. 
        # So we need to process kernel per output channel *per input channel*.

        # Unstack input channels dimension, then apply kernel constraint on each 2D slice:
        w_processed = []
        for kernel_slice in w_unstack:  # kernel_slice shape: (H, W, in_channels)
            # kernel_slice shape: (H, W, in_channels)
            # We unstack input channels (axis=-1)
            kernels_in = tf.unstack(kernel_slice, axis=-1)  # each is (H, W)
            kernels_in_processed = [ _kernel_constraint(k) for k in kernels_in ]
            # Stack back input channels
            stacked_in = tf.stack(kernels_in_processed, axis=-1)  # shape (H, W, in_channels)
            w_processed.append(stacked_in)

        # Stack back output channels
        w_final = tf.stack(w_processed, axis=-1)  # shape (H, W, in_channels, out_channels)
        return w_final


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers like in the example
        # Use the custom RadialConstraint in Conv2D kernel_constraint param.
        self.conv1 = tf.keras.layers.Conv2D(
            64, kernel_size=3, activation='relu',
            kernel_constraint=RadialConstraint(),
            input_shape=(28, 28, 1)
        )
        self.conv2 = tf.keras.layers.Conv2D(
            32, kernel_size=3, activation='relu'
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor matching MNIST shape: (batch_size, 28, 28, 1)
    # Batch size arbitrarily chosen as 8 here.
    return tf.random.uniform(shape=(8, 28, 28, 1), dtype=tf.float32)

