# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← assuming input shape (batch, height, width, channels) for TF model (converted from PyTorch NCHW)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Assumptions based on the PyTorch example provided in the issue:
        # Input shape in PyTorch was (1, 3, 32, 32)
        # Here we adapt to TensorFlow format NHWC: (B, 32, 32, 3)
        #
        # PyTorch model:
        # conv: Conv2d(3 -> 32, kernel=3)
        # relu
        # permute to NHWC for dense
        # dense1: Linear(32 -> 6) applied to channels dimension
        # flatten
        # dense2: Linear(6*(input_h-2)*(input_w-2) -> num_classes=10)
        # softmax
        #
        # We replicate this logic in TF with layers.
        #
        # Note: Because the first dense operates on channels dimension per spatial location,
        # we can implement it using a TimeDistributed Dense or Conv1D applied per spatial location.
        # Since input shape is NHWC, channels last, dense1 applies to last dim.
        
        self.conv = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, padding='valid', input_shape=(32, 32, 3))
        self.relu = tf.keras.layers.ReLU()
        
        # dense1 operates on channels dimension (last dim)
        # So apply Dense(6) to last dim of (B, H-2, W-2, 32)
        self.dense1 = tf.keras.layers.Dense(6)
        
        # After dense1 shape: (B, H-2, W-2, 6)
        # Flatten for dense2
        # dense2 output units = 10 (num_classes)
        self.flatten = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(10)
        
        self.softmax = tf.keras.layers.Softmax(axis=1)  # softmax over classes axis

    def call(self, x):
        # x shape: (B, 32, 32, 3)
        x = self.conv(x)  # (B, 30, 30, 32)
        x = self.relu(x)
        
        # dense1 applies the same dense layer on channels dimension at each spatial location
        # So we can use tf.keras.layers.TimeDistributed(Dense) over spatial dims
        # But TF does not have TimeDistributed for 2D spatial dims, instead we reshape to (-1, channels)
        # and apply dense, then reshape back.
        
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        
        # Reshape to (B*H*W, C)
        x_reshape = tf.reshape(x, (-1, C))
        x_dense1 = self.dense1(x_reshape)  # shape (B*H*W, 6)
        
        # Reshape back to (B, H, W, 6)
        x_dense1 = tf.reshape(x_dense1, (B, H, W, 6))
        
        x_flat = self.flatten(x_dense1)  # (B, H*W*6)
        x_out = self.dense2(x_flat)      # (B, 10)
        
        return self.softmax(x_out)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor of shape (1, 32, 32, 3) with dtype float32
    # to match the expected input of MyModel
    return tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)

