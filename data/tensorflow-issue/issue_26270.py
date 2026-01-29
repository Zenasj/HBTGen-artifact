# tf.random.uniform((B, T, C), dtype=tf.float32) ← Input shape inferred as (batch_size, timesteps, channels = 153)
import tensorflow as tf
from tensorflow.keras.layers import Layer

# We assume a global argument object for args.lstm_num as referenced in the example.
# Since it's not defined, let's define a placeholder for it.
class Args:
    lstm_num = 3  # Assumption: number of LSTM subnets is 3, based on example indexing

args = Args()

@tf.custom_gradient
def select_subnet_layer(x):
    """
    Custom operation that selects weighted subnet layers from input tensor x.

    Input shape: (batch_size, timesteps, channels)
    Assumes last args.lstm_num channels contain the selection weights.
    The remaining channels are divided evenly into lstm_num parts (subnets).

    Output shape: (batch_size, timesteps, size_of_out)
      where size_of_out = (channels - lstm_num) // lstm_num
    """
    # Shape info (static)
    batch_size = tf.shape(x)[0]
    timesteps = tf.shape(x)[1]
    total_channels = tf.shape(x)[2]

    # Compute size per subnet output
    size_of_out = (total_channels - args.lstm_num) // args.lstm_num

    # Split into select weights and data parts
    x_select = x[:, :, -args.lstm_num:]        # Shape: (B, T, lstm_num)
    x_data = x[:, :, :-args.lstm_num]          # Shape: (B, T, total_channels - lstm_num)

    # Start zero tensor for weighted sum output
    out = tf.zeros((batch_size, timesteps, size_of_out), dtype=x.dtype)

    # Weighted sum across subnets
    for i in range(args.lstm_num):
        # Get subnet slice
        subnet_slice = x_data[:, :, i*size_of_out:(i+1)*size_of_out]
        # Weight by corresponding selection channel
        weight = x_select[:, :, i:i+1]  # keep dims for broadcast
        out += subnet_slice * weight

    def custom_grad(dy):
        """
        Gradient function for select_subnet_layer.

        dy: upstream gradient of shape (B, T, size_of_out)

        Returns gradient of same shape as input x: (B, T, total_channels)
        Gradient w.r.t input combines gradients w.r.t x_data and x_select.

        The gradient for x_data parts is dy weighted by selection weights.
        The gradient for x_select parts is sum over subnet outputs weighted by dy.
        """
        # Gradient tensor shape
        grad = tf.TensorArray(dtype=x.dtype, size=args.lstm_num * 2)

        # Gradients w.r.t x_data parts
        grad_x_data_parts = []
        # Gradients w.r.t x_select parts
        grad_x_select_parts = []

        for i in range(args.lstm_num):
            # Gradient w.r.t x_data: upstream gradient dy * x_select weight
            grad_x_data_i = dy * x_select[:, :, i:i+1]  # shape (B, T, size_of_out)
            grad_x_data_parts.append(grad_x_data_i)

        for i in range(args.lstm_num):
            # Gradient w.r.t x_select: sum over (x_data * dy) across size_of_out dim
            subnet_slice = x_data[:, :, i*size_of_out:(i+1)*size_of_out]  # (B, T, size_of_out)
            grad_x_select_i = tf.reduce_sum(subnet_slice * dy, axis=2, keepdims=True)  # (B, T, 1)
            grad_x_select_parts.append(grad_x_select_i)

        # Concatenate gradients: first all x_data grads, then all x_select grads
        grad_x_data_concat = tf.concat(grad_x_data_parts, axis=2)       # (B, T, lstm_num * size_of_out)
        grad_x_select_concat = tf.concat(grad_x_select_parts, axis=2)   # (B, T, lstm_num)

        # Combine to form full gradient w.r.t x (channels dimension)
        grad_x = tf.concat([grad_x_data_concat, grad_x_select_concat], axis=2)  # (B, T, total_channels)

        return grad_x

    return out, custom_grad

class CustomLayer(Layer):
    """
    Keras layer wrapper around select_subnet_layer.

    Input: (batch_size, timesteps, channels)
    Output: (batch_size, timesteps, (channels - lstm_num) // lstm_num)
    """
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)

    def call(self, x):
        return select_subnet_layer(x)

    def compute_output_shape(self, input_shape):
        total_channels = input_shape[-1]
        size_of_out = (total_channels - args.lstm_num) // args.lstm_num
        return (input_shape[0], input_shape[1], size_of_out)

# The MyModel class that encapsulates this custom layer.
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Use the CustomLayer defined above
        self.custom_layer = CustomLayer()

    def call(self, inputs):
        # inputs expected to be tf.Tensor of shape (B, T, C)
        return self.custom_layer(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create a random input tensor matching the expected input shape:
    # Choose a batch size and time steps arbitrarily for example
    batch_size = 4
    time_steps = 10
    channels = 153  # consistent with example, channels including lstm_num selection channels

    # Generate a uniform random float32 tensor in range [0,1)
    x = tf.random.uniform(shape=(batch_size, time_steps, channels), dtype=tf.float32)

    # This matches the input expected by MyModel
    return x

