# tf.random.uniform((B, T), dtype=tf.int32) ← Inferred input shape: batch of token index sequences (B=batch size, T=sequence length)

import tensorflow as tf

class OneHotEmbedding(tf.keras.layers.Layer):
    """
    Replacement embedding layer that mimics tf.keras.layers.Embedding using 
    one-hot encoding and dot product. Used to illustrate memory behavior differences.
    """
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings = self.add_weight(
            shape=(input_dim, output_dim),
            initializer="uniform",
            trainable=True,
            name="embeddings"
        )

    def call(self, inputs):
        one_hot = tf.one_hot(inputs, depth=self.input_dim, dtype=tf.float32)
        return tf.linalg.matmul(one_hot, self.embeddings)


class MyModel(tf.keras.Model):
    """
    A simplified pseudo-Transformer style model for testing embedding memory behavior.
    Includes embedding layers for input and target vocabularies, basic encoder-decoder 
    style dense layers without positional encoding, masking, or residuals.

    Forward pass takes inputs:
      inputs: tf.int32 tensor shape (batch_size, seq_len) for input token IDs
      targets: tf.int32 tensor shape (batch_size, seq_len) for target token IDs (shifted)

    The model embeds inputs and targets separately, applies a simple dense layer, and 
    outputs logits over the target vocabulary.

    This mirrors the reuse of embedding layers in encoder and decoder and is the setup 
    used in the issue describing memory leak patterns.
    """

    def __init__(self, inp_vocab_size=8443, tar_vocab_size=8356,
                 embedding_dim=128, hidden_dim=256):
        super().__init__()
        # Embeddings - can be swapped with tf.keras.layers.Embedding or OneHotEmbedding
        # reflecting the reported workarounds / experiments
        self.input_embedding = tf.keras.layers.Embedding(inp_vocab_size, embedding_dim)
        self.target_embedding = tf.keras.layers.Embedding(tar_vocab_size, embedding_dim)

        # Simple encoder and decoder "layers" - single dense to reduce to hidden_dim
        self.encoder_dense = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.decoder_dense = tf.keras.layers.Dense(hidden_dim, activation="relu")

        # Output projection to target vocabulary logits
        self.output_layer = tf.keras.layers.Dense(tar_vocab_size)

    def call(self, inputs):
        """
        inputs: tuple (input_seq, target_seq)
          input_seq: (batch_size, T_inp) token ids
          target_seq: (batch_size, T_tar) token ids
        """
        input_seq, target_seq = inputs

        # Embed the sequences
        input_emb = self.input_embedding(input_seq)    # (B, T_inp, embedding_dim)
        target_emb = self.target_embedding(target_seq) # (B, T_tar, embedding_dim)

        # Encode input
        enc_out = self.encoder_dense(input_emb)  # (B, T_inp, hidden_dim)

        # Decode target
        dec_out = self.decoder_dense(target_emb)  # (B, T_tar, hidden_dim)

        # For simplicity, combine encoder and decoder outputs by broadcasting input encoding
        # here just concatenate and flatten - this is not a real transformer but reflects 
        # the simplified setup from the issue reproducible code.
        # Since sequence lengths can differ, a simple approach here:
        combined = tf.concat([enc_out, dec_out], axis=1)  # (B, T_inp + T_tar, hidden_dim)

        # Produce logits for each timestep over target vocab
        logits = self.output_layer(combined)  # (B, T_inp + T_tar, tar_vocab_size)

        return logits


def my_model_function():
    """
    Creates an instance of MyModel with typical vocab sizes from the issue.
    Embedding dims and hidden dims are chosen reasonably.
    """
    # These vocab sizes and dims are taken from the issue
    inp_vocab_size = 8443
    tar_vocab_size = 8356
    embedding_dim = 128
    hidden_dim = 256

    model = MyModel(inp_vocab_size, tar_vocab_size, embedding_dim, hidden_dim)
    return model


def GetInput():
    """
    Generates a dummy input matching expected model input signature:
    a tuple (input_seq, target_seq), each a (batch_size, seq_length) tensor of token IDs.

    Using batch_size=64 and sequence length=37 as common values inferred from the OOM logs.
    Token IDs are uniformly sampled in vocab size.
    """
    batch_size = 64
    seq_length = 37

    # Same vocab sizes as used in model
    inp_vocab_size = 8443
    tar_vocab_size = 8356

    # Random integer input sequences in [0, vocab_size)
    input_seq = tf.random.uniform(
        shape=(batch_size, seq_length), minval=0, maxval=inp_vocab_size, dtype=tf.int32
    )
    target_seq = tf.random.uniform(
        shape=(batch_size, seq_length), minval=0, maxval=tar_vocab_size, dtype=tf.int32
    )
    return (input_seq, target_seq)

