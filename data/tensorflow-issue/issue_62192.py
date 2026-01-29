# tf.random.uniform((B, ), dtype=tf.float32) for inputs a: (None, 3), b: (None, 4), c: (None, 5)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        # Single Dense layer as in original example
        self.sublayer = tf.keras.layers.Dense(16)

    def call(self, x, **kwargs):
        # Based on the Keras 3 recommendation, inputs are expected at same level, not nested.
        # x is a dict with keys: 'a', 'b', 'c'
        a = x["a"]
        b = x["b"]
        c = x["c"]
        # Concatenate along last dimension
        concat_tensors = tf.concat([a, b, c], axis=-1)
        return self.sublayer(concat_tensors)

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Generate a dictionary of inputs matching expected shapes and dtype
    # batch size is chosen arbitrarily as 2 for demonstration
    batch_size = 2
    a = tf.random.uniform((batch_size, 3), dtype=tf.float32)
    b = tf.random.uniform((batch_size, 4), dtype=tf.float32)
    c = tf.random.uniform((batch_size, 5), dtype=tf.float32)
    return {"a": a, "b": b, "c": c}

