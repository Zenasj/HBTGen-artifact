# tf.random.uniform((256, 12, 20), dtype=tf.int32) ← inferred input shape based on issue example

import tensorflow as tf
from tensorflow.keras import layers as tfl

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer with mask_zero=True as in the original example
        self.embed_layer = tfl.Embedding(input_dim=4500, output_dim=32, mask_zero=True)
        # MultiHeadAttention with 2 heads, attention_axes=2 (attending over last spatial dim)
        self.attn_layer = tfl.MultiHeadAttention(num_heads=2, key_dim=16, attention_axes=2)

    def call(self, x):
        """
        x: shape (batch=256, seq_len=12, spatial=20)
        After embedding: (256, 12, 20, 32)
        """
        # Embedding the input
        x = self.embed_layer(x)  # (256, 12, 20, 32), mask_zero=True produces a mask internally

        # Original issue suggests a bug occurs when using MultiHeadAttention with attention mask and attention_axes.
        # A mentioned workaround is concatenating the tensor along axis=0,
        # but that doesn't fully make sense here and would change batch size.
        # Instead, to build a model compatible and runnable for testing,
        # we omit that and just directly call MHA as the user code does.

        # Directly calling attention, query/key/value are x with shape (256, 12, 20, 32).
        # This produces output with same shape.
        x = self.attn_layer(query=x, key=x, value=x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random integer tensor consistent with Embedding input, shape (256, 12, 20)
    # Values between 0 and 4499 (vocab size - 1)
    return tf.random.uniform((256, 12, 20), minval=0, maxval=4499, dtype=tf.int32)

