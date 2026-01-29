# tf.random.uniform((B, max_len), dtype=tf.string) ← 
# Input consists of two parts: 
# 1) token strings input of shape (batch_size, max_len)
# 2) handcrafted features per token of shape (batch_size, max_len, 40)

import tensorflow as tf
import tensorflow_hub as hub

# We assume elmo_model is loaded once globally for the class.
# Use batch_size and max_len as passed parameters or fixed for signature.
# To avoid graph/control-flow errors shown in the original issue,
# we implement ELMo embedding extraction inside a tf.keras.Layer that
# safely calls the signature.

class ElmoEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, elmo_model, max_len, **kwargs):
        super().__init__(**kwargs)
        self.elmo_model = elmo_model
        self.max_len = max_len

    def call(self, x):
        # x is shape (batch_size, max_len) with dtype tf.string
        # tokens signature expects tokens=[batch_size, max_len], sequence_len=[batch_size]
        # We cast to tf.string in case of dtype issues
        tokens = tf.cast(x, tf.string)
        batch_size = tf.shape(tokens)[0]
        sequence_len = tf.fill([batch_size], self.max_len)
        # Use the 'tokens' signature of elmo_model
        # It returns a dict with key "elmo" of shape (batch_size, max_len, 1024)
        return self.elmo_model.signatures["tokens"](tokens=tokens, sequence_len=sequence_len)["elmo"]

class MyModel(tf.keras.Model):
    def __init__(self, max_len, n_words, n_tags, elmo_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.n_tags = n_tags
        self.elmo_layer = ElmoEmbeddingLayer(elmo_model, max_len, name="elmo_embedding")
        
        # Layers for handcrafted features branch
        # Input shape: (batch_size, max_len, 40)
        self.word_dense = tf.keras.layers.Dense(n_tags, activation='softmax', name="word_dense")

        # After concatenating ELMo embedding and dense word features
        self.concat = tf.keras.layers.Concatenate(axis=-1, name="concat_features")
        self.batchnorm = tf.keras.layers.BatchNormalization(name="batch_norm")
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2
            ),
            name="bilstm"
        )
        self.timedist_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(n_tags, activation='softmax'), name="time_dist_dense"
        )

    def call(self, inputs, training=False):
        # inputs is a list or tuple: [elmo_input_layer, word_input_layer]
        # elmo_input_layer: (batch_size, max_len), dtype=tf.string tokens
        # word_input_layer: (batch_size, max_len, 40) handcrafted features
        elmo_input, word_input = inputs
        elmo_embeds = self.elmo_layer(elmo_input)  # (batch_size, max_len, 1024)
        word_out = self.word_dense(word_input)     # (batch_size, max_len, n_tags)
        
        x = self.concat([word_out, elmo_embeds])   # (batch_size, max_len, n_tags + 1024)
        x = self.batchnorm(x, training=training)
        x = self.bilstm(x, training=training)      # (batch_size, max_len, 1024)
        output = self.timedist_dense(x, training=training)  # (batch_size, max_len, n_tags)
        return output

def my_model_function():
    # We load the ELMo TF Hub module once, here at model init
    # IMPORTANT: The URL below is the TF Hub ELMo module for TF2 compatible usage
    elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
    
    # Hypothetical parameters; user must set these properly
    max_len = 50  # Max tokens in a sentence
    n_words = 10000  # Vocabulary size unused directly here
    n_tags = 17  # Number of output tags/classes for token classification
    
    return MyModel(max_len, n_words, n_tags, elmo_model)

def GetInput():
    # Returns a tuple of inputs consistent with MyModel call
    # 1) elmo_input_layer: strings shape (batch_size, max_len)
    # 2) word_input_layer: float32 features shape (batch_size, max_len, 40)
    batch_size = 2
    max_len = 50
    feature_dim = 40

    # Generate random "tokens" as zero-padded dummy strings like "word1", "word2", ...
    # To simulate tokenized sentences. We keep consistent batch size and max_len.
    tokens = tf.strings.as_string(tf.random.uniform([batch_size, max_len], maxval=10000, dtype=tf.int32))
    
    # Some handcrafted features, random float values
    features = tf.random.uniform([batch_size, max_len, feature_dim], dtype=tf.float32)
    
    return (tokens, features)

