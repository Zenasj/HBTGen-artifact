# tf.random.uniform((B, H, W, C), dtype=...) ← Input shape is assumed to be (batch, height, width, channels) as typical for InstanceNorm usage.

import tensorflow as tf
from tensorflow import keras

# Custom InstanceNormalization layer implementing stable mixed precision computations.
class InstanceNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-3, **kwargs):
        # Force dtype to float32 for numerical stability as discovered in the issue.
        # This avoids variance overflow in mixed precision training.
        kwargs['dtype'] = tf.float32
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Scale and offset are learned parameters, 1D with shape matching channels.
        channel_dim = input_shape[-1]
        # Initialize scale close to 1, offset to 0
        self.scale = self.add_weight(
            name='scale',
            shape=(channel_dim,),
            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02),
            trainable=True,
            dtype=tf.float32,
        )
        self.offset = self.add_weight(
            name='offset',
            shape=(channel_dim,),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32,
        )
        # Normalize over spatial dimensions (height and width)
        self.axis = list(range(1, len(input_shape) - 1))
        super().build(input_shape)

    def call(self, x):
        # Cast input to float32 for stable mean and variance computation
        x_float32 = tf.cast(x, tf.float32)

        # Compute mean and variance over spatial dims (instance norm)
        mean = tf.reduce_mean(x_float32, axis=self.axis, keepdims=True)
        variance = tf.reduce_mean(tf.square(x_float32 - mean), axis=self.axis, keepdims=True)

        # Normalize with epsilon for numerical stability
        normalized = (x_float32 - mean) / tf.sqrt(variance + self.epsilon)

        # Reshape scale and offset for broadcasting: (1,1,1,C)
        scale_reshaped = tf.reshape(self.scale, [1] * (len(x.shape) - 1) + [-1])
        offset_reshaped = tf.reshape(self.offset, [1] * (len(x.shape) - 1) + [-1])

        # Apply scale and offset in float32
        y = normalized * scale_reshaped + offset_reshaped

        # Cast back to input dtype (possibly float16) to preserve mixed precision context of the rest of model
        return tf.cast(y, x.dtype)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # This MyModel is a minimal example showing usage of the InstanceNormalization layer
        # in a typical CNN block with mixed precision.

        # Assume input shape: (batch, height, width, channels)
        # We use float16 for inputs (typical in mixed precision),
        # but InstanceNormalization internals use float32 for numeric stability.

        # Simple Conv2D layer before norm
        self.conv = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding='same',
            dtype=tf.float16,
            kernel_initializer='he_normal',
        )

        # Our custom InstanceNormalization layer forcing internal computations to float32
        self.inst_norm = InstanceNormalization()

        # Activation layer
        self.activation = keras.layers.Activation('relu', dtype=tf.float16)

    def call(self, inputs):
        # inputs expected to be float16 tensor with shape [B,H,W,C]

        # Pass through Conv2D in float16
        x = self.conv(inputs)

        # InstanceNorm layer inside performs internal cast to float32, then casts back output to float16
        x = self.inst_norm(x)

        # Activation in float16
        x = self.activation(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Create a random input tensor with shape [batch, height, width, channels]
    # Using batch=4, height=64, width=64, channels=3 (e.g. RGB image)
    # dtype float16 to simulate mixed precision input (typical scenario)
    input_shape = (4, 64, 64, 3)
    return tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float16)

