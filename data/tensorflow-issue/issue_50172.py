# tf.random.uniform((1, 244, 244, 3), dtype=tf.float32) ← fixed batch size 1, input shape 244x244 with 3 channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, dilation_rate=(2, 2)):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            dilation_rate=dilation_rate,
            use_bias=False
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        
    def call(self, inputs, training=False):
        # Conv2d -> BatchNorm -> ReLU pattern as described
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x

def my_model_function():
    # Return an instance of MyModel with fixed dilation as per issue (dilation=2)
    return MyModel(dilation_rate=(2, 2))

def GetInput():
    # Generate a fixed batch size input tensor compatible with the model input.
    # Batch size =1, 244x244 spatial dims, 3 channels, float32 as per the example code.
    return tf.random.uniform((1, 244, 244, 3), dtype=tf.float32)

