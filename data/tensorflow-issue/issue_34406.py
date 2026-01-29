# tf.random.uniform((B, 28, 28, 1), dtype=tf.float16) ← input shape inferred from FashionMNIST example

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Activation
)
from tensorflow.keras.models import Sequential

class MyModel(tf.keras.Model):
    """
    This model implements the FashionMNIST classifier architecture discussed,
    using mixed precision policy 'mixed_float16' with special handling for
    the final softmax layer for numeric stability and compatibility.

    Internally:
    - Input shape: (batch_size, 28, 28, 1)
    - Uses Conv2D + MaxPooling layers with relu activations in mixed_float16.
    - Final Dense softmax layer is computed in float32 for numeric stability.
    """

    def __init__(self):
        super().__init__()
        # Set mixed precision policy for computation
        # Using tf.keras.mixed_precision.experimental.Policy since TensorFlow 2.0
        # This policy will cause computation to use float16 internally where possible,
        # with some ops (like softmax) in float32 for stability.
        self.mixed_precision_policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(self.mixed_precision_policy)

        # Define model layers consistent with the example in the issue
        self.conv1 = Conv2D(32, (3, 3), activation='relu', name='conv2d_1')
        self.pool1 = MaxPooling2D((2, 2), name='maxpool_1')

        self.conv2 = Conv2D(64, (3, 3), activation='relu', name='conv2d_2')
        self.pool2 = MaxPooling2D((2, 2), name='maxpool_2')

        self.conv3 = Conv2D(64, (3, 3), activation='relu', name='conv2d_3')

        self.glob_avg_pool = GlobalAveragePooling2D(name='global_avg_pool')

        self.dense1 = Dense(64, activation='relu', name='dense_1')

        # Final Dense layer outputs logits in float16 (due to policy),
        # but softmax activation is separated and forced to run in float32 for stability.
        self.final_dense = Dense(10, name='dense_logits')  # no activation here
        self.softmax = Activation('softmax', dtype='float32', name='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.glob_avg_pool(x)

        x = self.dense1(x)

        logits = self.final_dense(x)
        # Softmax in float32 for numeric stability while allowing upstream mixed_float16 performance
        out = self.softmax(logits)
        return out

def my_model_function():
    """
    Returns an instance of MyModel with mixed precision policy set.
    """
    # Setting the policy globally here too is helpful if used outside MyModel class
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    return MyModel()

def GetInput():
    """
    Returns a random tensor input matching the input expected by MyModel:
    batch size = 32 (arbitrary choice)
    height = 28, width = 28, channels = 1
    dtype = float16 due to mixed precision policy
    """
    batch_size = 32
    input_shape = (batch_size, 28, 28, 1)
    # Input dtype is float16 for mixed precision training
    return tf.random.uniform(input_shape, dtype=tf.float16)

