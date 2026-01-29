# tf.random.uniform((B, T), dtype=tf.int32)  # B=batch_size, T=sequence_length, input is integer tokens for Embedding

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=1000, embed_dim=16, sequence_length=20, arch='max_pool'):
        super().__init__()
        # Embedding layer with mask_zero=True to enable masking for padded tokens (index 0)
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embed_dim,
                                                   mask_zero=True,
                                                   name='embedding')

        # Architecture choice: either a layer that supports masking or does not
        # GlobalAvgPool1D supports masking, Flatten and GlobalMaxPool1D do not
        self.arch = arch
        if arch == 'avg_pool':
            self.pooling = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool_1d')
            self.supports_masking = True
        elif arch == 'max_pool':
            self.pooling = tf.keras.layers.GlobalMaxPooling1D(name='global_max_pool_1d')
            self.supports_masking = False
        elif arch == 'flatten':
            self.pooling = tf.keras.layers.Flatten(name='flatten')
            self.supports_masking = False
        else:
            # Default to average pooling if unknown arch provided
            self.pooling = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool_1d')
            self.supports_masking = True

    def call(self, inputs, training=None):
        x = self.embedding(inputs)  # shape: (batch, seq_len, embed_dim)
        mask = self.embedding.compute_mask(inputs)  # mask has shape (batch, seq_len)

        # Check mask support: raise error if layer does not support masking but mask is present
        # This enforces the behavior that the official issue describes as missing:
        # raising an error when using a layer that does not support masking after a masking layer.
        if mask is not None and not self.supports_masking:
            # Raise informative error similar to TF 1.x behavior:
            raise TypeError(
                f"Layer {self.pooling.name} does not support masking, "
                "but was passed an input mask. "
                "Consider removing mask propagation or choose a masking-compatible layer."
            )

        # If layer supports masking, pass the mask so internal layers can handle it correctly
        if self.supports_masking:
            # Some layers like GlobalAveragePooling1D use the mask automatically if passed
            # However, Keras layers typically use input masks implicitly when set properly,
            # so we can rely on this automatic behavior.
            output = self.pooling(x)
        else:
            # If no masking support, mask will be ignored internally
            output = self.pooling(x)

        return output

def my_model_function(arch='max_pool'):
    """
    Returns an instance of MyModel with the specified arch type.

    Args:
        arch: str, one of {'max_pool', 'avg_pool', 'flatten'}. Determines which pooling/flattening layer to use after Embedding.

    Returns:
        MyModel: a compiled model instance.
    """
    return MyModel(arch=arch)

def GetInput(batch_size=4, sequence_length=20, vocab_size=1000):
    """
    Returns a random integer tensor input suitable for feeding into MyModel.
    Generates random sequences of token IDs in [1, vocab_size), to avoid the padding index 0.

    Args:
        batch_size: int, number of sequences in batch.
        sequence_length: int, length of each sequence.
        vocab_size: int, vocabulary size, must match embedding input_dim.

    Returns:
        tf.Tensor of shape (batch_size, sequence_length) with dtype tf.int32
    """
    # Generate random token indices between 1 and vocab_size-1
    # Index 0 is reserved for padding and triggers masking in Embedding
    input_tensor = tf.random.uniform(
        shape=(batch_size, sequence_length),
        minval=1,
        maxval=vocab_size,
        dtype=tf.int32
    )
    return input_tensor

