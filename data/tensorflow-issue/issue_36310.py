# tf.random.uniform((B, T, H, W, C), dtype=tf.float32)
# Here: B=batch size, T=sequence length (time steps), H=height=640, W=width=480, C=channels=1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the model structure from the issue:
        # MaxPooling3D layer with pool size (1, 2, 2), input shape (None, 640, 480, 1)
        # ConvLSTM2D with 16 filters, kernel 3x3
        # Flatten
        # Dense 1 unit
        
        # MaxPooling3D (pool_size=(1,2,2)) expects 5D input: (batch, depth, height, width, channels)
        # The input shape in the issue is (None, 640, 480, 1) for each time step dimension.
        # Since ConvLSTM2D expects 5D input: (batch, time, rows, cols, channels),
        # and MaxPooling3D also works on 5D tensors, the pool size's first dim 1 pools over time dimension with no change.

        self.max_pool3d = tf.keras.layers.MaxPooling3D(pool_size=(1,2,2))
        self.conv_lstm2d = tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(3,3))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        Args:
            inputs: Tensor of shape (batch_size, time_steps, height=640, width=480, channels=1)
        Returns:
            output Tensor of shape (batch_size, 1)
        """
        # Apply MaxPooling3D layer first (same as in the issue code)
        x = self.max_pool3d(inputs)
        # Then ConvLSTM2D layer
        x = self.conv_lstm2d(x)
        # Flatten and Dense
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    """
    Returns an instance of MyModel.
    Weights are randomly initialized as no pretrained weights were specified.
    """
    return MyModel()

def GetInput():
    """
    Returns a random input tensor that matches the input expected by MyModel:
    Shape: (batch_size, time_steps, height, width, channels) → (1, 10, 640, 480, 1)
    This matches the example test input used in the issue reproduction code.

    Using tf.random.uniform with dtype=tf.float32.
    """
    batch_size = 1
    time_steps = 10
    height = 640
    width = 480
    channels = 1
    inp = tf.random.uniform(shape=(batch_size, time_steps, height, width, channels), dtype=tf.float32)
    return inp

