# tf.random.uniform((1, 5, 6, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters as described in the issue
        filters = 2
        kernel_size = [3, 3]
        strides = [1, 1]
        padding = "same"
        output_padding = None
        data_format = "channels_last"
        dilation_rate = [2, 2]
        activation = "linear"
        use_bias = True

        # Initialize Conv2DTranspose layer with given parameters
        self.conv2d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            dtype=tf.float32,
        )

    def call(self, inputs):
        """
        Forward pass through conv2d_transpose layer.

        Additionally computes:
         - res_backward: output of layer(inputs)
         - grad_backward: jacobian of res_backward w.r.t. res_backward (expected to be identity-ish)
         - grad_jvp: jacobian-vector-product via ForwardAccumulator on res_forward
        
        Returns a dictionary of tensors for comparison.
        """
        # Compute the layer output
        res_backward = self.conv2d_transpose(inputs)

        # We compute the jacobian of output w.r.t itself - this is a very large tensor,
        # shape: (batch, height_out, width_out, filters, batch, height_out, width_out, filters)
        # According to the issue, grad_backward shape was (1,5,6,2,1,5,6,2)
        with tf.GradientTape(persistent=True) as g:
            g.watch(res_backward)
            # Pass-through identity to keep trace of res_backward
            identity_out = tf.identity(res_backward)
        grad_backward = g.jacobian(identity_out, res_backward)

        # Using ForwardAccumulator to get JVP (Jacobian-vector product)
        tangents = tf.constant(1.0, dtype=tf.float32, shape=res_backward.shape)
        with tf.autodiff.ForwardAccumulator(res_backward, tangents) as acc:
            res_forward = self.conv2d_transpose(inputs)
            grad_jvp = acc.jvp(res_forward)

        # Return outputs for comparison as a dictionary
        return {
            "res_backward": res_backward,
            "grad_backward": grad_backward,
            "res_forward": res_forward,
            "grad_jvp": grad_jvp,
        }


def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor compatible with MyModel's Conv2DTranspose layer input
    # Shape according to the issue: [1, 5, 6, 1]
    return tf.random.uniform([1, 5, 6, 1], minval=-2, maxval=2, dtype=tf.float32)

