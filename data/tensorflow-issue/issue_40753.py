# tf.random.uniform((B, 1), dtype=tf.float32)  ← Input shape inferred from toy_dataset inputs: tf.range(10.)[:, None]
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A simple linear layer with 5 outputs, matching the original Net Dense(5)
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        # Forward pass: linear layer on input x
        return self.l1(x)

def my_model_function():
    # Return an instance of MyModel (same as the original Net model)
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape by MyModel
    # Original input shape is (batch_size, 1), toy_dataset used batches of size 2 in example
    batch_size = 2
    # Using uniform random floats similar to tf.range input but random for generality
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)

