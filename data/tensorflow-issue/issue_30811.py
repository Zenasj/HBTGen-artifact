# tf.random.uniform((1, 64, 64, 8), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # BatchNormalization layer as described in the issue
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, train_bn=False):
        # Forward pass using BatchNormalization layer with explicit training flag
        return self.bn(x, training=train_bn)

def my_model_function():
    # Returns an instance of MyModel with BatchNormalization initialized
    return MyModel()

def GetInput():
    # Return a random tensor input matching the input signature [1, 64, 64, 8], dtype float32
    # Matching the input signature used in the tf.function for inference in the issue
    return tf.random.uniform((1, 64, 64, 8), dtype=tf.float32)

