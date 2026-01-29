# tf.random.uniform((B, 24, 24, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Basic Conv2D layer with tanh activation fused into the model
        self.conv = tf.keras.layers.Conv2D(10, kernel_size=1)
        # Note:
        # Due to known TensorFlow Model Optimization Toolkit limitations,
        # tanh activation is not currently supported directly in quantization aware training.
        # To work around this, we separate the conv and tanh layers and
        # apply fake_quant manually after tanh as shown in the referenced workaround.

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        # Apply tanh activation manually after convolution
        x = tf.nn.tanh(x)
        # Insert a fake quantization node to simulate quantization behavior on the tanh output
        # This simulates int8 quantization range typically between -1 and 1 for tanh
        x = tf.quantization.fake_quant_with_min_max_args(
            x, min=-1.0, max=1.0, num_bits=8, narrow_range=False, name=None)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching the input shape expected by MyModel
    # Batch size B is chosen arbitrarily as 1 here
    B = 1
    H = 24
    W = 24
    C = 3
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)

