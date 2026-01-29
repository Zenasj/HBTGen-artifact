# tf.random.uniform((1, 5, 4), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We replicate the tf.compat.v1.layers.MaxPooling1D call with large pool_size to show behavior.
        # Because tf.compat.v1.layers.MaxPooling1D is deprecated, we simulate MaxPooling1D with tf.keras.layers.
        # Note: Large pool_size (like 1e38) will cause errors in CuDNN backend when running on GPU.
        # Here, pool_size and strides are forced as in the bug report.
        # pool_size = [1e+38] is unrealistically large and causes cudnnSetPoolingNdDescriptor failure.
        # For demonstration, we accept float to int truncation to int(1e38) which is huge (will cause error).
        self.pool_size = int(1e+38)
        self.strides = 2
        self.padding = "same"
        self.data_format = "channels_last"

        # Create a 1D max pooling layer with the configured parameters.
        # Use tf.keras.layers.MaxPooling1D directly instead of deprecated tf.compat.v1 layer.
        # We catch errors that arise on GPU due to large pool size.
        self.max_pool = tf.keras.layers.MaxPooling1D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

    def call(self, inputs, training=None):
        # The large pool_size will likely cause error on GPU (CuDNN failure), replicate the issue.
        # We wrap in try-except here for defensive programming but in standard model call,
        # errors propagate normally. We just do it to note the error scenario.
        try:
            out = self.max_pool(inputs)
        except Exception as e:
            # Instead of crashing, return the error string as a tensor for demonstration.
            # In real usage, unhandled exceptions occur (as reported).
            err_str = tf.constant("Error: " + str(e))
            # Convert string to numeric tensor not possible, so return a tensor of shape (1,) with 0.
            # User can check outputs outside this model for string info.
            out = tf.zeros((1, 1, 1), dtype=inputs.dtype)
            tf.print(err_str)
        return out

def my_model_function():
    # Return an instance of MyModel with the given initialization
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape: (1, 5, 4)
    # batch size=1, sequence length=5, channels=4, dtype float32
    return tf.random.uniform((1, 5, 4), dtype=tf.float32)

