# tf.random.uniform((100, 10), dtype=tf.float32) ← Input shape inferred from Dataset batch (batch_size=100, feature_dim=10)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Rebuild the original sequential model structure
        self.dense1 = tf.keras.layers.Dense(256)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel; weights are uninitialized here
    # For actual use, load weights accordingly or train first
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape expected by MyModel: batch_size=100, features=10, dtype float32
    return tf.random.uniform((100, 10), dtype=tf.float32)

