# tf.random.uniform((B, max_len), dtype=tf.string)  ← Inferred ELMo input shape: batch of token sequences (strings)
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Bidirectional, LSTM, TimeDistributed, Lambda
from tensorflow.keras.models import Model

# Note:
# The original issue is about integrating ELMo embeddings from TF Hub in TF2 using hub.load().
# The provided code tries to use elmo_model.signatures["tokens"] inside a Lambda layer,
# which triggers an AttributeError during training due to TF2 autograph / control flow issues with hub.load().
# The original poster's workaround was to fallback to TF1's hub.Module, since hub.load() does not seamlessly support this use case in TF2.
#
# Here, we reconstruct the model as a Keras subclassed model to reflect the same architecture,
# but we encapsulate the ELMo embedding call as a submodule that requires an external hub handle.
# Because full seamless ELMo usage via TF2 hub.load() signatures inside training is problematic,
# this class expects the ELMo layer/function to be injected or overridden.
#
# We provide a placeholder ELMo embedding function returning zeros, to allow testing and compilation.
# In practice, users need to implement or wrap ELMo embeddings appropriately,
# potentially as a frozen TF1 graph or precomputed embeddings.
#
# The input shapes and architecture are preserved:
# - elmo_input_layer: shape (max_len,) with dtype string (tokens)
# - word_input_layer: shape (max_len, 40) float features per token
#
# The model concatenates a dense softmax projection on word_input_layer (per token),
# with the ELMo embeddings per token (size 1024),
# then applies batch normalization, BiLSTM, and final time-distributed softmax layer.
#
# The forward call returns token-level predictions of shape (batch_size, max_len, n_tags).

class MyModel(tf.keras.Model):
    def __init__(self, max_len, n_tags, elmo_layer=None):
        super().__init__()
        self.max_len = max_len
        self.n_tags = n_tags
        # The dense layer applied on the additional word features
        self.word_dense = Dense(n_tags, activation='softmax')
        # Batch normalization after concatenation
        self.batchnorm = BatchNormalization()
        # BiLSTM with recurrent dropout
        self.bilstm = Bidirectional(
            LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)
        )
        # Final time distributed dense layer for tag prediction
        self.final_dense = TimeDistributed(Dense(n_tags, activation='softmax'))

        # ELMo embedding layer/function which maps (B, max_len) string tokens -> (B, max_len, 1024)
        if elmo_layer is None:
            # Placeholder: returns zeros for ELMo embeddings.
            # Replace this with actual ELMo embedding function or TF Hub layer.
            self.elmo_layer = Lambda(
                lambda x: tf.zeros(shape=(tf.shape(x)[0], self.max_len, 1024), dtype=tf.float32)
            )
        else:
            self.elmo_layer = elmo_layer

    def call(self, inputs, training=False):
        # inputs is a tuple/list: (elmo_input, word_features)
        elmo_input, word_features = inputs
        # elmo_input shape: (batch_size, max_len) dtype string
        # word_features shape: (batch_size, max_len, 40) float

        # Pass word_features through dense softmax per token
        word_proj = self.word_dense(word_features)  # (B, max_len, n_tags)

        # Pass tokens through the ELMo embedding layer/function
        # Should produce (B, max_len, 1024)
        elmo_emb = self.elmo_layer(elmo_input)   # Expect elmo_emb shape for concatenation

        # Concatenate along the last axis the word_proj and ELMo embeddings
        x = Concatenate(axis=-1)([word_proj, elmo_emb])  # shape: (B, max_len, n_tags + 1024)

        x = self.batchnorm(x, training=training)
        x = self.bilstm(x, training=training)
        output = self.final_dense(x)  # (B, max_len, n_tags) softmax per token

        return output

def my_model_function(max_len=300, n_tags=17, n_words=None, elmo_layer=None):
    """
    Factory function to instantiate MyModel.

    Args:
      - max_len (int): maximum sequence length (number of tokens)
      - n_tags (int): number of tag labels for classification per token
      - n_words: not used here but kept for API compatibility
      - elmo_layer: optional callable or tf.keras.Layer to embed tokens with ELMo, 
        expected input shape (B, max_len), dtype string; output shape (B, max_len, 1024)
        If None, a placeholder zero embedding layer is used.

    Returns:
      - MyModel instance
    """
    model = MyModel(max_len=max_len, n_tags=n_tags, elmo_layer=elmo_layer)
    return model

def GetInput(max_len=300):
    """
    Returns example inputs to feed the model.

    Returns:
      - Tuple of (elmo_input_tokens, word_features)
        elmo_input_tokens: tf.Tensor of dtype string, shape (batch_size, max_len)
        word_features: tf.Tensor of dtype float32, shape (batch_size, max_len, 40)
    """
    batch_size = 32  # Example batch size
    # Random string tokens: for demo purposes, use random integers as strings
    tokens = tf.random.uniform(
        shape=(batch_size, max_len), maxval=10000, dtype=tf.int32
    )
    # Convert to strings
    elmo_input_tokens = tf.strings.as_string(tokens)

    # Random float features for each token (shape (B, max_len, 40))
    word_features = tf.random.uniform(
        shape=(batch_size, max_len, 40), dtype=tf.float32
    )

    return (elmo_input_tokens, word_features)

