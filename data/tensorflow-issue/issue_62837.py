# tf.random.uniform((B, T), dtype=tf.int32)  # B=batch size, T=sequence length (token ids input to embedding)

import tensorflow as tf

def positional_encoding(length, depth):
    """Generate a positional encoding tensor with shape (length, depth)."""
    # This is a standard sine-cosine positional encoding implementation,
    # commonly used in transformer models such as GPT.
    depth = depth // 2
    positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]  # (length, 1)
    depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth  # (1, depth)
    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (length, depth)

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)  # (length, 2*depth)
    return tf.cast(pos_encoding, dtype=tf.float32)  # (length, depth*2)

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=10000, d_model=512, max_len=2048):
        """
        A simplified model that implements PositionalEmbedding as seen in the issue.
        - vocab_size: size of vocabulary for embedding.
        - d_model: model dimension for embedding vectors.
        - max_len: maximum sequence length for positional encodings.
        
        This follows the reported code in the issue, including
        the scaling by sqrt(d_model) which was causing JIT issues in some setups.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(max_len, d_model)  # precomputed pos encodings
    
    def call(self, x):
        """
        Input:
          x -- integer tensor of shape (batch_size, sequence_length), token IDs
        
        Output:
          embedded tensor of shape (batch_size, sequence_length, d_model)
          with scaled embeddings plus positional encodings added.
        """
        length = tf.shape(x)[1]
        x = self.embedding(x)  # (B, T, d_model)
        # Multiply embeddings by sqrt(d_model) as scaling factor (as in the original code).
        # This is the line that triggered GPU jit compilation errors in TF 2.15,
        # but works normally with CUDA driver upgrade as per the issue resolution.
        scale = tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x * scale
        # Add positional encoding slice up to input length
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

def my_model_function():
    # Instantiate MyModel with example vocab size and d_model matching the issue example
    # Using the defaults: vocab_size=10000, d_model=512, max_len=2048
    return MyModel()

def GetInput():
    """
    Generate a random input tensor compatible with MyModel.
    - Output shape: (batch_size, sequence_length), integer token ids.
    - Values in [1, vocab_size-1] because mask_zero=True means 0 is reserved for padding mask.
    
    For demonstration, batch_size=2, sequence_length=10 is chosen arbitrarily.
    """
    batch_size = 2
    sequence_length = 10
    vocab_size = 10000  # Should match MyModel's default vocab_size
    # Generate random token ids in [1, vocab_size-1]
    return tf.random.uniform(
        (batch_size, sequence_length),
        minval=1,
        maxval=vocab_size,
        dtype=tf.int32
    )

