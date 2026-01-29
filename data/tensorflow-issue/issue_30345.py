# tf.random.uniform((B, 4, 4, 3), dtype=tf.float32) ← input shape inferred from original model input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Upsampling layer: Conv2DTranspose that upsamples 4x4x3 input to 8x8x3 output
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=4, strides=2, padding='same')

        # Flatten layer before Dense is required to avoid reshape errors in skip connection
        self.flatten = tf.keras.layers.Flatten()

        # Dense layer outputting 8*8*3 features to be reshaped as 8x8x3
        self.dense = tf.keras.layers.Dense(8 * 8 * 3)

        # Reshape layer to convert dense output to (8,8,3)
        self.reshape = tf.keras.layers.Reshape([8, 8, 3])

        # Add layer to sum conv_transpose output and linear skip output
        self.add = tf.keras.layers.Add()

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs: [B, 4, 4, 3]
        # Apply Conv2DTranspose to upsample
        y = self.conv_transpose(inputs)  # shape [B, 8, 8, 3]

        # Flatten then Dense then Reshape branch for skip connection
        linear = self.flatten(inputs)     # shape [B, 4*4*3=48]
        linear = self.dense(linear)       # shape [B, 8*8*3=192]
        linear = self.reshape(linear)     # shape [B, 8, 8, 3]

        # Add the two tensors elementwise
        out = self.add([y, linear])       # shape [B, 8, 8, 3]
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random float32 tensor with shape (batch=8, 4,4,3) matching model input
    return tf.random.uniform(shape=(8, 4, 4, 3), dtype=tf.float32)

