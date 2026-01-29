# tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model with one Dense layer to match the original example
        self.dense = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # The example input shape is (batch, 1), inferred from dataset with shape (1,)
    # Use batch size 10 (as in the original example batches of 10)
    # dtype float32 to match typical TF default
    batch_size = 10
    input_shape = (batch_size, 1)
    # Generate random float input tensor
    return tf.random.uniform(input_shape, dtype=tf.float32)

