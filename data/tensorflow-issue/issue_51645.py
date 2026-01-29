# tf.random.uniform((B, 2*C), dtype=tf.float32) ← Input shape inferred as single tensor of shape (?, features)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Layers as per original example: Concatenate followed by Dense(10, relu)
        self.con = tf.keras.layers.Concatenate()  # Concatenates on last axis by default
        self.dense = tf.keras.layers.Dense(10, activation="relu")

    def call(self, x):
        # Concatenate input tensor with itself along the last axis
        x = self.con([x, x])
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Input must match the expected input shape of MyModel's call method.
    # Since the dense layer has units=10, input should be (batch, features).
    # Original example does not specify input shape, but Concatenate doubles feature dimension.
    # To keep things flexible, choose input shape (batch=4, features=5).
    batch_size = 4
    features = 5
    # Generate random float32 tensor of shape (batch_size, features)
    return tf.random.uniform((batch_size, features), dtype=tf.float32)

