# tf.random.uniform((B, 3, 8, 8, 8), dtype=tf.float32) ← inferred typical input shape with channels_first (NCDHW)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 3D convolution layer with channels_first data format
        self.conv3d = tf.keras.layers.Conv3D(
            filters=12,
            kernel_size=(2, 2, 2),
            data_format="channels_first",
            use_bias=False,
            padding="valid",
            strides=(1, 1, 1),
        )
    
    def call(self, x):
        # Apply Conv3D with channels_first format
        return self.conv3d(x)


def my_model_function():
    # Return an instance of MyModel with Conv3D channels_first configured
    return MyModel()


def GetInput():
    # Generate random input tensor matching shape (batch=1, channels=3, depth=8, height=8, width=8)
    # dtype float32 as expected by Conv3D
    return tf.random.uniform((1, 3, 8, 8, 8), dtype=tf.float32)

