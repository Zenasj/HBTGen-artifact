# tf.random.uniform((B, input_vocab_size, d_model), dtype=tf.float32)

import tensorflow as tf

def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class TestModel(tf.keras.Model):
    def __init__(self, d_model=128, input_vocab_size=233, target_vocab_size=666):
        super().__init__()
        self.d_model = d_model
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        self.embedding_inp = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.embedding_tar = tf.keras.layers.Embedding(target_vocab_size, d_model)

        # Positional encoding tensor initialized as a random uniform tensor.
        # Shape: [1, input_vocab_size, d_model]
        # This mimics the static positional encoding tensor from the original code.
        self.pos_encoding_inp = tf.random.uniform([1, input_vocab_size, d_model])

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar):
        # WARNING: Using inp.shape[1] (static shape) can cause errors due to
        # shape mismatch between dynamic batch inputs and static positional encoding.
        # Using tf.shape(inp)[1] (dynamic shape) is correct and prevents errors.
        # Here we intentionally use inp.shape[1] to demonstrate the error mentioned.

        # Using static shape (can cause errors in certain TF versions/models)
        seq_len_static = inp.shape[1]

        # Using dynamic shape (preferred to avoid shape mismatch errors)
        seq_len_dynamic = tf.shape(inp)[1]

        # Choose to use dynamic shape here as that is the TensorFlow recommended practice
        seq_len = seq_len_dynamic

        inp_emb = self.embedding_inp(inp)  # shape: (batch_size, seq_len, d_model)
        # Add positional encoding slice - must slice using dynamic seq_len to match batch input
        inp_emb += self.pos_encoding_inp[:, :seq_len, :]

        tar_emb = self.embedding_tar(tar)  # (batch_size, seq_len_tar, d_model)

        attn_out, _ = scaled_dot_product_attention(q=tar_emb, k=inp_emb, v=inp_emb)

        out = self.final_layer(attn_out)  # (batch_size, seq_len_tar, target_vocab_size)
        return out

class MyModel(tf.keras.Model):
    def __init__(self, d_model=128, input_vocab_size=233, target_vocab_size=666):
        super().__init__()
        # Encapsulate the TestModel logic inside MyModel

        self.test_model = TestModel(d_model, input_vocab_size, target_vocab_size)

    def call(self, inputs):
        # inputs expected to be a tuple (inp, tar)
        inp, tar = inputs
        # Forward through the encapsulated model
        output = self.test_model(inp, tar)
        return output

def my_model_function():
    # Return an instance of MyModel with default parameters
    return MyModel()

def GetInput():
    # Return a tuple of input tensors matching MyModel's expected input signature:
    # inp: shape (batch_size, seq_len_inp), dtype int64
    # tar: shape (batch_size, seq_len_tar), dtype int64
    # Use typical values from the example code:
    batch_size = 64
    seq_len_inp = 40
    seq_len_tar = 39
    # Vocab sizes should be consistent with those in TestModel defaults
    input_vocab_size = 233
    target_vocab_size = 666

    import numpy as np
    inp = tf.constant(
        np.random.randint(0, input_vocab_size, size=(batch_size, seq_len_inp)),
        dtype=tf.int64)
    tar = tf.constant(
        np.random.randint(0, target_vocab_size, size=(batch_size, seq_len_tar)),
        dtype=tf.int64)
    return (inp, tar)

