# tf.random.uniform((B, 75), dtype=tf.int32) ← input shape: batch of sequences of length 75 (token IDs)

import tensorflow as tf
import numpy as np

# Since `tokenizer`, `embedding_matrix`, and flags like `use_embedding` and `emb_trainable` 
# are mentioned externally and used in model construction, I will define them as placeholders 
# for this example, with typical assumptions:
# - vocab_size is dynamic as len(tokenizer.word_index)+1
# - embedding_dim = 50 (from GloVe example)
# Assuming embedding usage True, embeddings trainable False.

embedding_dim = 50
max_len = 75
use_embedding = True
emb_trainable = False

# For tokenizer simulation, suppose vocab size is 10000:
vocab_size = 10000  # hypothetical vocab size
tokenizer_word_index_size = vocab_size - 1  # tokenizer.word_index assumed to have vocab_size - 1 tokens

# Create a dummy embedding matrix filled with zeros (real use should load GloVe or trained embeddings)
embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

# Custom LuongAttention layer re-implemented for TF 2.20.0 compatibility
class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, input_dim=max_len, att_type='dot', **kwargs):
        super(LuongAttention, self).__init__(**kwargs)
        self.att_type = att_type
        self.input_dim = input_dim
        w_init = tf.random.normal
        if att_type == 'general':
            # Weight matrix for scoring general attention
            self.W = self.add_weight(
                shape=(input_dim, input_dim),
                initializer=w_init,
                trainable=True,
                name='W_general')
        else:
            # For 'dot' or other types, we either use identity or Dense layer
            self.W = tf.eye(input_dim)
        # Dense layer for 'concat' type attention if implemented (not used here)
        self.WLayer = tf.keras.layers.Dense(input_dim, 
                                           kernel_regularizer=tf.keras.regularizers.l2(0.01))
        # Loss regularization term on W for training
        self.loss_scale = 1e-5

    def call(self, inputs):
        # inputs: list or tuple of two tensors: [query, values]
        q, v = inputs
        # Add L2 loss regularization on the weight matrix
        self.add_loss(self.loss_scale * tf.nn.l2_loss(self.W))
        if self.att_type == 'concat':
            # (Not used in final model code but implemented for completeness)
            score = tf.matmul(q, self.WLayer(v), transpose_b=True)
            alignment = tf.nn.softmax(score, axis=2)
        else:
            # 'dot' and 'general' types
            # score = q * W * v^T
            # Note: W shape is (input_dim, input_dim)
            # q shape: (batch_size, time, input_dim)
            # v shape: (batch_size, time, input_dim)
            score = tf.matmul(v, self.W)  # -> (batch_size, time, input_dim)
            score = tf.matmul(q, score, transpose_b=True)  # -> (batch_size, time_q, time_v)
            alignment = tf.nn.softmax(score, axis=2)  # attention weights
        return alignment

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer setup
        if use_embedding:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                input_length=max_len,
                trainable=emb_trainable,
                name='embedding')
        else:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_len,
                trainable=True,
                name='embedding')

        # Bidirectional LSTM layers with ELU activation and L2 regularization
        self.bi_lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(25, activation='elu', return_sequences=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            name='bi_lstm1')
        self.bi_lstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(25, activation='elu', return_sequences=True,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            name='bi_lstm2')
        # Two Attention layers - here the final code uses standard scaled dot product Attention
        self.attention1 = tf.keras.layers.Attention(use_scale=True, name='attention1')
        self.attention2 = tf.keras.layers.Attention(use_scale=True, name='attention2')
        # Concatenate the two attention outputs on last axis
        self.concat = tf.keras.layers.Concatenate(axis=-1, name='concat')
        # Another Bidirectional LSTM (return_sequences=False by default)
        self.bi_lstm3 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(80, activation='elu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            name='bi_lstm3')
        # Final Dense output with sigmoid (scalar output)
        self.dense_out = tf.keras.layers.Dense(1, activation='sigmoid', name='output')

    def call(self, inputs):
        # inputs is a list or tuple: [in1, in2], each shape (batch_size, max_len)
        in1, in2 = inputs
        # Apply embedding to both inputs
        emb1 = self.embedding(in1)  # (batch_size, max_len, embedding_dim)
        emb2 = self.embedding(in2)

        # Run bidir LSTM layers on embedded inputs
        out1 = self.bi_lstm1(emb1)  # (batch_size, max_len, 50)
        out2 = self.bi_lstm2(emb2)  # (batch_size, max_len, 50)

        # Attention layers - note: attention query and value inputs
        attn1 = self.attention1([emb1, emb2])  # attention between base embeddings
        attn2 = self.attention2([out1, out2])  # attention between LSTM encoded sequences

        # Concatenate both attentions on last axis
        cat = self.concat([attn1, attn2])  # (batch_size, ?, 100) shape

        # Process concatenated attention with another Bidirectional LSTM
        out = self.bi_lstm3(cat)  # (batch_size, 160)

        # Final output scalar in range [0,1]
        output = self.dense_out(out)  # (batch_size, 1)

        return output

def my_model_function():
    # Return an instance of MyModel (weights are randomly initialized)
    return MyModel()

def GetInput():
    # Returns valid random input compatible with MyModel:
    # Two sequences of integers in [0, vocab_size), shape (batch_size, max_len)
    batch_size = 4
    in1 = tf.random.uniform((batch_size, max_len), minval=1, maxval=vocab_size, dtype=tf.int32)
    in2 = tf.random.uniform((batch_size, max_len), minval=1, maxval=vocab_size, dtype=tf.int32)
    return [in1, in2]

