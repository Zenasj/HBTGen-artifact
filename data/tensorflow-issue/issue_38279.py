# tf.random.uniform((1, 1, 16, 16, 1), dtype=tf.float32)
import tensorflow as tf

class CustomCellConv2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # State size matches the conv2d output spatial and channel dimensions
        self.state_size = tf.TensorShape((16, 16, 1))
        super(CustomCellConv2D, self).__init__(**kwargs)
        # Kernel is fixed as ones(3,3,1,1)
        self.kernel = tf.ones((3, 3, 1, 1), dtype=tf.float32)

    def call(self, inputs, states, **kwargs):
        # inputs shape: (batch, height, width, channels) = (1, 16, 16, 1)
        # states[0] shape: (batch, height, width, channels)
        conv = tf.nn.conv2d(inputs, self.kernel, strides=(1, 1), padding="SAME")
        output = states[0] + conv
        new_state = output
        return output, new_state

class CustomCellConv1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # State size matches the conv1d output shape (16, 1)
        self.state_size = tf.TensorShape((16, 1))
        super(CustomCellConv1D, self).__init__(**kwargs)
        # Kernel fixed as ones(3,1,1)
        self.kernel = tf.ones((3, 1, 1), dtype=tf.float32)

    def call(self, inputs, states, **kwargs):
        # inputs shape: (batch, length, channels) = (1, 16, 1)
        conv = tf.nn.conv1d(inputs, self.kernel, stride=1, padding="SAME")
        output = states[0] + conv
        new_state = output
        return output, new_state

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # RNN with CustomCellConv2D cell
        self.rnn_conv2d = tf.keras.layers.RNN(
            CustomCellConv2D(), 
            return_sequences=False,
            return_state=False,
            # Note: unroll=False by default, causes grappler cycle error in TF 2.1+
        )
        # RNN with CustomCellConv1D cell
        self.rnn_conv1d = tf.keras.layers.RNN(
            CustomCellConv1D(), 
            return_sequences=False,
            return_state=False,
        )

    def call(self, inputs):
        """
        Takes a 5D tensor input shaped for conv2d RNN: (batch, time, H, W, channels)
        Also derives 3D input for conv1d RNN by squeezing W dimension.
        Compares the outputs approximately (they live in different shapes, so comparison is illustrative).
        Returns a dict with both outputs and a boolean indicating if outputs match within tolerance.
        
        Assumptions:
        - The conv2d input is (1,1,16,16,1)
        - For conv1d, inputs reshaped from (1,1,16,16,1) to (1,1,16,1)
        - Comparison tolerance is set loosely since shapes and ops differ.
        """
        # inputs shape: (batch, time, 16, 16, 1)
        # conv2d expects (batch, time, H, W, channels)
        output2d = self.rnn_conv2d(inputs)

        # For conv1d cell, reshape inputs: combine W dimension and channel dimension as channels=1 with length=16
        # Extract: shape (batch, time, 16, 1)
        inputs_1d = tf.reshape(inputs, shape=(tf.shape(inputs)[0], tf.shape(inputs)[1], 16, 1))
        output1d = self.rnn_conv1d(inputs_1d)

        # Compare the outputs for approximate equality by broadcasting:
        # output2d shape: (batch, 16, 16, 1)
        # output1d shape: (batch, 16, 1)
        # Broadcast output1d to (batch,16,16,1)
        output1d_broadcast = tf.broadcast_to(output1d, tf.shape(output2d))

        # Compute element-wise difference
        diff = tf.abs(output2d - output1d_broadcast)

        # Define a tolerance threshold
        tolerance = 1e-3

        # Check if all differences are within tolerance
        outputs_equal = tf.reduce_all(diff < tolerance)

        # Return a structured output
        return {
            "output_conv2d": output2d,
            "output_conv1d_broadcast": output1d_broadcast,
            "outputs_equal": outputs_equal,
            "diff": diff,
        }

def my_model_function():
    """
    Instantiate MyModel.
    """
    return MyModel()

def GetInput():
    """
    Return a 5D tensor of shape (1, 1, 16, 16, 1) with float32 values,
    compatible as input to MyModel.
    """
    # As per the original reported input shape and dtype
    return tf.ones((1, 1, 16, 16, 1), dtype=tf.float32)

