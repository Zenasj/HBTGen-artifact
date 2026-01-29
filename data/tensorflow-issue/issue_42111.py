# tf.random.uniform((3, 496, 496, 64), dtype=tf.float32) ← Input batch of 3 images, each 496x496 with 64 channels

import tensorflow as tf

# Workaround from the issue: add an explicit numerical dependency to enforce ordering of SyncBatchNorm layers
def dependency(x, y):
    """Returns an instance of x that nominally depends on y (but is numerically identical to x).
    This enforces execution ordering both forward and backward.
    """
    return x + 0 * tf.reduce_mean(y)

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(name="my_model", **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding="SAME")
        self.syncbn1 = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)
        self.syncbn2 = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding="SAME")
        self.syncbn3 = tf.keras.layers.experimental.SyncBatchNormalization(axis=-1)
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=False):
        # First conv + SyncBatchNorm
        conv1 = self.conv1(inputs)
        conv1bn = self.syncbn1(conv1, training=training)

        # Second SyncBatchNorm, on output of first BN
        deconv1bn = self.syncbn2(conv1bn, training=training)

        # Enforce dependency so conv2 and further BN execute after deconv1bn
        conv1bn_dep = dependency(conv1bn, deconv1bn)

        # Convolution using the dependency-enforced tensor
        conv2 = self.conv2(conv1bn_dep)
        conv2bn = self.syncbn3(conv2, training=training)

        # Concatenate all three batch norm outputs
        return self.concat([conv1bn, deconv1bn, conv2bn])

def my_model_function():
    model = MyModel()
    # Normally weights are initialized on first call; no external weights to load here
    return model

def GetInput():
    # Generate a batch of 3 random images with shape (496, 496, 64) matching expected input
    return tf.random.uniform(shape=(3, 496, 496, 64), dtype=tf.float32)

