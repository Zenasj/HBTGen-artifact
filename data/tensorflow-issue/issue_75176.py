# tf.random.uniform((128, 80), dtype=tf.int32)  ← Input shape inferred from batch_size=128 and max_review_len=80

import tensorflow as tf

batch_size = 128
total_words = 10000
max_review_len = 80
embedding_len = 100
units = 64  # inferred from original code's usage

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Embedding layer maps word indices to embedding vectors
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        # Two SimpleRNNCell layers with dropout
        self.RNNCell0 = tf.keras.layers.SimpleRNNCell(units, dropout=0.2)
        self.RNNCell1 = tf.keras.layers.SimpleRNNCell(units, dropout=0.2)
        # Final dense layer to produce single logit output
        self.outlayer = tf.keras.layers.Dense(1)
        # Note: states are not stored as fixed attributes to avoid inaccessible tensor error,
        # they will be initialized per batch in call()

    @tf.function(jit_compile=True)
    def call(self, inputs, training=None):
        """
        inputs: int32 tensor shape (batch_size, max_review_len),
                word indices for sequences
        """
        x = self.embedding(inputs)  # (batch_size, max_review_len, embedding_len)

        # Initialize the states as zeros each call to avoid InaccessibleTensorError
        state0 = [tf.zeros((tf.shape(inputs)[0], units), dtype=x.dtype)]
        state1 = [tf.zeros((tf.shape(inputs)[0], units), dtype=x.dtype)]

        # Unstack along time dimension to get list of (batch_size, embedding_len) tensors
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.RNNCell0(word, states=state0, training=training)
            out1, state1 = self.RNNCell1(out0, states=state1, training=training)

        x = self.outlayer(out1)
        prob = tf.sigmoid(x)
        return prob

def my_model_function():
    # Return an instance of MyModel with initialized layers
    return MyModel()

def GetInput():
    # Return random input tensor of shape (batch_size, max_review_len)
    # with word indices from 0 to total_words - 1
    # dtype int32 as required by Embedding
    return tf.random.uniform(
        (batch_size, max_review_len), minval=0, maxval=total_words, dtype=tf.int32)

