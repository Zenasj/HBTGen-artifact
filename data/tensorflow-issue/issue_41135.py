# tf.random.uniform((B, None), dtype=tf.int32) ← Input is batch-size by variable length sequences of token ids

import tensorflow as tf

class MyWordEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyWordEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # This weight matrix maps token IDs (0..299) to 512-dim embeddings
        self.kernel = self.add_weight(
            shape=(300, 512),
            dtype='float32',
            initializer='glorot_uniform',
            trainable=True)
        super(MyWordEmbedding, self).build(input_shape)

    def call(self, inputs):
        # inputs: integer token IDs, shape (batch_size, seq_len)
        # Use tf.nn.embedding_lookup: maps token IDs to embedding vectors
        # Expect inputs shape: (batch_size, seq_len)
        tokens = inputs  # expecting shape (B, None)
        return tf.nn.embedding_lookup(params=self.kernel, ids=tokens)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape[0] expected: (batch_size, seq_len, embedding_dim=512)
        # Qdense is a weight matrix (512 x 512)
        self.Qdense = self.add_weight(
            name='Qdense',
            shape=(512, 512),
            dtype='float32',
            initializer='glorot_uniform',
            trainable=True)
        super(EncoderLayer, self).build(input_shape)

    def call(self, inputs):
        """
        inputs: Tuple of two tensors
          - inputs[0]: embedded inputs, shape (batch_size, seq_len, 512)
          - inputs[1]: mask parameter float tensor, shape (batch_size, seq_len)
        """
        x, mask_para = inputs

        # Linear projections
        Qoutput = tf.einsum('aij,jk->aik', x, self.Qdense)
        Koutput = tf.einsum('aij,jk->aik', x, self.Qdense)
        Voutput = tf.einsum('aij,jk->aik', x, self.Qdense)

        # Compute scaled dot-product style attention scores modulated by mask_para
        # a shape after einsum is (batch_size, seq_len, seq_len)
        # mask_para is shaped (batch_size, seq_len).
        # tf.tile to match attention dims: tile axis=1 (query seq length) with size 64 as fixed sequence length
        # We must assume max seq_len=64 here as per original use case, 
        # but since input seq_len is dynamic, safe assumption is None or max 64.
        # Because this is a demo, we follow original shape fixed tiling of 64.

        # Expand mask_para dims and tile as (batch_size, 64, seq_len)
        # This matches shape of a after einsum for elementwise multiplication
        mask_tiled = tf.tile(
            tf.expand_dims(mask_para, axis=1), 
            multiples=[1, 64, 1])

        a = tf.einsum('ajk,afk->ajf', Qoutput, Koutput) * mask_tiled
        a = tf.matmul(a, Voutput)  # final attention output shape: (batch_size, 64, 512)

        return a

    def compute_mask(self, inputs, mask):
        # Pass masks through unchanged
        return mask

    def compute_output_shape(self, input_shape):
        # output shape same as input embeddings (batch, seq_len, 512)
        return input_shape[0]


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Instantiate submodules
        self.embedding = MyWordEmbedding()
        self.encoder = EncoderLayer()

    def call(self, inputs):
        """
        inputs: tuple or list of two elements:
          - inputs[0]: word_ids_en, shape (batch_size, seq_len_en), int32 token ids for decoder input
          - inputs[1]: word_ids_fr, shape (batch_size, seq_len_fr), int32 token ids for encoder input
          
        The model will:
          - Embed the English input (decoder side)
          - Generate mask and embedding for French input (encoder side)
          - Pass encoder embedding and mask to EncoderLayer, get encoder output
          - Add encoder output to decoder embedding and return result.
        """
        word_ids_en, word_ids_fr = inputs

        # Embed French input and get mask to pass to encoder
        embedded_fr = self.embedding(word_ids_fr)  # (B, seq_len_fr, 512)
        mask_fr = tf.cast(tf.not_equal(word_ids_fr, 0), dtype='float32')  # (B, seq_len_fr)
        
        encoded = self.encoder([embedded_fr, mask_fr])  # (B, seq_len_fr, 512)

        # Embed English input
        embedded_en = self.embedding(word_ids_en)  # (B, seq_len_en, 512)
        
        # We want to add encoder output to decoder embedding.
        # They must have compatible shapes.
        # For simplicity, assume seq_len_en == seq_len_fr == For example 64 in original padding.
        # If seq lengths differ, this add may error; 
        # in real use case, usually they match or further processing applied.
        output = embedded_en + encoded  # shape broadcasted elementwise add

        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate random inputs consistent with the expected input:
    # Two integer tensors (word_ids_en, word_ids_fr)
    # Each shaped (batch_size, seq_len), dtype int32
    # Tokens from vocabulary range [0, 299)
    batch_size = 3
    seq_len = 64
    vocab_size = 300

    word_ids_en = tf.random.uniform(
        shape=(batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)
    word_ids_fr = tf.random.uniform(
        shape=(batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)

    return (word_ids_en, word_ids_fr)

