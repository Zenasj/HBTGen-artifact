# tf.random.uniform((B, 5), dtype=tf.float32) ← inferred input shape from provided example (batch_size, features)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Typical BatchNormalization layer for feature size 5 as per example
        self.bn = tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, x, training=None):
        # Important to pass `training` to BN layer to distinguish train/inference mode
        x = self.bn(x, training=training)
        return x

def my_model_function():
    # Return an instance of MyModel.
    # Users can compile/train externally as needed.
    return MyModel()

def GetInput():
    # Return a random float32 tensor input matching the input expected by MyModel:
    # The example data was shape (10, 5), so batch size can vary; choosing 10 as per example
    return tf.random.uniform((10, 5), dtype=tf.float32)

