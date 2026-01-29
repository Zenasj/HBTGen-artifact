# tf.random.uniform((B, 84, 84, 3), dtype=tf.float32) ← Input shape inferred from issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original example: a Conv2D -> BatchNorm(momentum=0.0) -> ReLU -> Flatten -> Dense(1)
        self.conv = tf.keras.layers.Conv2D(10, 3, activation=None)
        # BatchNormalization with momentum=0.0 to reflect the scenario described in the issue
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        self.relu = tf.keras.layers.ReLU()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        """
        Forward pass. Supports training and inference mode.
        
        Note: The reported issue concerns BatchNorm with momentum=0.0 during training=True,
        which in some TF versions raises ValueError if mean and variance are not None.
        This code reflects expected usage, where moment=0 (no running average)
        means the layer uses current batch statistics exclusively during training.
        """
        x = self.conv(inputs)
        # Explicitly pass training flag to BatchNorm layer
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.flatten(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Instantiate and return the MyModel instance
    return MyModel()

def GetInput():
    # Return a random float32 tensor of shape (batch_size=4, height=84, width=84, channels=3)
    # Batch size 4 is chosen since the original issue example runs training on batch of 4 inputs
    return tf.random.uniform(shape=(4, 84, 84, 3), dtype=tf.float32)

