# tf.random.uniform((B, None), dtype=tf.int32)  ← Input is a batch of variable-length integer sequences (word IDs)

import tensorflow as tf
from tensorflow.keras import backend as K

class MyWordEmbedding(tf.keras.layers.Layer):
    def build(self, input_shape):
        # Embedding kernel: vocab_size=300, embedding_dim=512
        self.kernel = self.add_weight(shape=(300, 512), dtype='float32',
                                      initializer='glorot_uniform',
                                      name='embedding_kernel')
        super(MyWordEmbedding, self).build(input_shape)  # Call at end
    
    def call(self, inputs):
        # inputs expected as a single tensor (word IDs)
        # Use embedding lookup on the first element if inputs is a list/tuple,
        # but in our fused model, always a tensor directly.
        if isinstance(inputs, (list, tuple)):
            ids = inputs[0]
        else:
            ids = inputs
        return tf.nn.embedding_lookup(params=self.kernel, ids=ids)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # The mask_para tensor will be passed during call, not at init
        super(EncoderLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Qdense weight matrix (512x512)
        self.Qdense = self.add_weight(name='Qdense',
                                      shape=(512, 512),
                                      dtype='float32',
                                      initializer='glorot_uniform')
        super(EncoderLayer, self).build(input_shape)

    def call(self, x):
        # x is a list/tuple: [input_embeddings, mask_para] with
        # input_embeddings shape: (batch, seq_len, 512)
        # mask_para shape: (batch, seq_len)
        input_embeddings, mask_para = x

        # Compute Q, K, V by multiplying input embeddings by Qdense weight
        Qoutput = tf.einsum('aij,jk->aik', input_embeddings, self.Qdense)
        Koutput = tf.einsum('aij,jk->aik', input_embeddings, self.Qdense)
        Voutput = tf.einsum('aij,jk->aik', input_embeddings, self.Qdense)

        # Broadcast mask_para and apply attention-like scoring
        # mask_para shape (batch, seq_len)
        # expand dims at axis=1 to (batch, 1, seq_len)
        # tile to (batch, 64, seq_len) — assuming max seq length 64 is needed
        # NOTE: max seq_len is variable, so tile dynamically:
        seq_len = tf.shape(Qoutput)[1]
        mask_expanded = tf.expand_dims(mask_para, axis=1)  # (batch, 1, seq_len)
        mask_tiled = tf.tile(mask_expanded, [1, seq_len, 1])  # (batch, seq_len, seq_len)

        # Attention weights:
        a = tf.einsum('ajk,afk->ajf', Qoutput, Koutput) * mask_tiled
        # Matrix multiply attention weights by Voutput
        a = tf.matmul(a, Voutput)
        return a

    def compute_mask(self, inputs, mask):
        # Pass through the mask unchanged
        return mask

    def compute_output_shape(self, input_shape):
        # Output shape same as input_embeddings shape
        return input_shape[0]

class MyModel(tf.keras.Model):
    """
    Combined model encapsulating encoder and decoder logic,
    demonstrating the error scenario and proper call signatures.
    Inputs:
      - word_ids_en: int32 tensor shape (batch, seq_len)
      - word_ids_fr: int32 tensor shape (batch, seq_len)
    Forward Steps:
      1. Encode French word IDs -> embeddings -> encoded tensor via EncoderLayer.
      2. Decode English word IDs + encoded tensor embeddings, combined by addition,
         through embedding layer.
    Output:
      Combined tensor after addition (shape: batch, variable seq_len, 512)
    """
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding_layer = MyWordEmbedding()
        self.encoder_layer = EncoderLayer()

    def call(self, inputs):
        # inputs is a tuple/list: (word_ids_en, word_ids_fr)
        word_ids_en, word_ids_fr = inputs

        # Encoder path:
        # Embed French inputs
        embedded_fr = self.embedding_layer(word_ids_fr)  # (batch, seq_len_fr, 512)
        # Construct mask: cast not-equal-zero of French inputs (batch, seq_len_fr)
        mask_fr = tf.cast(tf.not_equal(word_ids_fr, 0), dtype='float32')
        # Apply EncoderLayer: pass [embeddings, mask]
        encoded_fr = self.encoder_layer([embedded_fr, mask_fr])  # (batch, seq_len_fr, 512)

        # Decoder path:
        # Embed English inputs
        embedded_en = self.embedding_layer(word_ids_en)  # (batch, seq_len_en, 512)

        # We need to add embedded_en + encoded_fr
        # Because seq_len of en and fr can differ, 
        # assume they are equal or truncate/pad as needed.
        # For simplicity, if seq_len differs, truncate to min seq_len
        seq_len_en = tf.shape(embedded_en)[1]
        seq_len_fr = tf.shape(encoded_fr)[1]
        min_seq_len = tf.minimum(seq_len_en, seq_len_fr)
        embedded_en_trimmed = embedded_en[:, :min_seq_len, :]
        encoded_fr_trimmed = encoded_fr[:, :min_seq_len, :]

        combined = embedded_en_trimmed + encoded_fr_trimmed  # (batch, min_seq_len, 512)
        return combined

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of inputs: (word_ids_en, word_ids_fr)
    # Both are int32 tensors with shape (batch_size, seq_len)
    # Use batch size 3, seq_len 64 (from example)
    batch_size = 3
    seq_len = 64

    # Random sequences of integers from 0 to 4 (like vocab indices)
    word_ids_en = tf.random.uniform((batch_size, seq_len), minval=0, maxval=5, dtype=tf.int32)
    word_ids_fr = tf.random.uniform((batch_size, seq_len), minval=0, maxval=5, dtype=tf.int32)

    return (word_ids_en, word_ids_fr)

