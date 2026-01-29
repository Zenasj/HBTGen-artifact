# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, label_num=1000, input_shape=(224, 224, 3)):
        super().__init__()
        # Input layer does not need to be explicitly defined in subclass model
        # Conv2D layer as per the example squeezenet snippet
        self.conv = tf.keras.layers.Conv2D(
            filters=96,
            kernel_size=(7, 7),
            strides=2,
            padding="valid",
            activation="relu",
            input_shape=input_shape,
        )
        # Cropping2D layer, allowing negative cropping values as per the discussion,
        # which would lead to zero-padding on that side.
        # Here we keep as in example (-100,) which will pad zeros on all sides.
        self.cropping = tf.keras.layers.Cropping2D(cropping=-100)
        # Dense layer applied after flattening is done in the example:
        # But the original code applied Dense directly on 4D tensor which is unconventional,
        # likely a mistake or typo in the issue report.
        # We will replicate the exact logic for fidelity:
        self.dense = tf.keras.layers.Dense(units=label_num, activation="softmax")
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.cropping(x)
        # Following original code logic: dense layer is applied before flattening,
        # which means dense is acting as a 1x1 conv (time-distributed dense), but dense expects 2D inputs.
        # To match example behavior, we apply dense on last axis per pixel:
        # So we reshape: merge height and width dims to batch dimension, apply dense, then reshape back
        # This is an inference to reproduce reported behavior though unusual.
        # Shape: (batch, H, W, channels) -> (batch*H*W, channels)
        shape = tf.shape(x)
        batch, h, w, c = shape[0], shape[1], shape[2], shape[3]
        x_reshaped = tf.reshape(x, (batch * h * w, c))
        x_dense = self.dense(x_reshaped)
        # Now reshape back to (batch, h, w, label_num)
        x = tf.reshape(x_dense, (batch, h, w, -1))
        x = self.flatten(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default params same as example
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected Input shape: (batch, 224,224,3)
    # Batch size can be arbitrary, let's choose 2
    return tf.random.uniform((2, 224, 224, 3), dtype=tf.float32)

