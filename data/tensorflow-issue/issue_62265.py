# tf.random.normal([9, 10, 3], dtype=tf.float32) ← input images shape for saturation adjustment
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Pre-generate random images once to ensure stable input across calls,
        # as pointed in the issue comments to avoid randomness inside the call method.
        self.images = tf.random.normal([9, 10, 3], dtype=tf.float32)

    @tf.function(jit_compile=True)
    def call(self, x):
        # Apply tf.raw_ops.AdjustSaturation with scale=x and fixed images.
        # Note: scale=x is a scalar float.
        out = tf.raw_ops.AdjustSaturation(scale=x, images=self.images)
        return out

def my_model_function():
    # Return an instance of MyModel; no additional initialization needed here.
    return MyModel()

def GetInput():
    # Return a random scalar float tensor for 'scale' to feed into model.
    # The scale parameter controls how much to adjust saturation.
    # Based on example, use scalar float32 shape [].
    return tf.random.normal([], dtype=tf.float32)

