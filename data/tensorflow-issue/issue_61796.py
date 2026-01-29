# tf.random.uniform((B, T, D), dtype=tf.float32) for query and value inputs
# Here, B=1 (batch size), T=variable sequence length (8 for query, 4 for value), D=16 (feature dimension)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # MultiHeadAttention layer with 2 heads and key dimension 2, 
        # matching the example setup (num_heads=2, key_dim=2)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)

    def call(self, inputs):
        """
        Expects `inputs` as a tuple: (q, v)
          - q: query tensor of shape (batch, 8, 16)
          - v: value tensor of shape (batch, 4, 16)
        Returns output of MultiHeadAttention layer.
        """
        q, v = inputs
        # Call the MHA layer with query and value both provided.
        # Key is not explicitly provided, so defaults to value.
        attn_output = self.mha(query=q, value=v)
        return attn_output

def my_model_function():
    # Build and return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of two random float32 tensors matching the input:
    # Query shape: (1, 8, 16)
    # Value shape: (1, 4, 16)
    # Using tf.random.uniform to simulate input data similar to original example
    q = tf.random.uniform(shape=(1, 8, 16), dtype=tf.float32)
    v = tf.random.uniform(shape=(1, 4, 16), dtype=tf.float32)
    return (q, v)

