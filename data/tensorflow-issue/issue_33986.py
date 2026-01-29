# tf.random.uniform((B, 20), dtype=tf.float32) ← Input shape inferred from Dense layer input_shape=(20,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple 2-layer model analogous to the example in the issue
        # 1 Dense layer with 1 unit, input shape (20,)
        self.dense = tf.keras.layers.Dense(1, input_shape=(20,))
        # Sigmoid activation layer
        self.activation = tf.keras.layers.Activation(tf.sigmoid)

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        return self.activation(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching (batch_size, input_dim)
    # Since batch size is ambiguous, assume B=4 for demonstration
    B = 4
    input_shape = (B, 20)
    # Use float32 dtype consistent with Keras defaults
    return tf.random.uniform(input_shape, dtype=tf.float32)

