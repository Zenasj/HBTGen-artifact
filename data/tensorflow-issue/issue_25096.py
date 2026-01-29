# tf.random.uniform((B, 5), dtype=tf.float32)  ← Input shape inferred from input_shape=[5]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dense layer with units=1
        # Followed by a softplus activation implemented using a standard callable (tf.nn.softplus)
        # This avoids the error when Keras tries to use a Layer instance as an activation
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense(inputs)
        # Use tf.nn.softplus directly, as recommended in the issue discussion for stable usage
        return tf.nn.softplus(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor of shape (batch_size, 5) with float32 dtype
    # Batch size here is arbitrarily chosen as 4 for demonstration
    return tf.random.uniform((4, 5), dtype=tf.float32)

