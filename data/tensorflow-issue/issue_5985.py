# tf.random.uniform((BATCH, MAX_LEN, EMBEDDING_DIM), dtype=tf.float32) ← inferred input shape after embedding_lookup (BATCH=5, MAX_LEN=10, EMBEDDING_DIM=300)

import tensorflow as tf

# Constants inferred from original code
BATCH = 5  # batch size
MAX_LEN = 10  # max length of the sequence
MLP_HIDDEN_DIM = 128  # number of hidden neurons in the MLP
EMBEDDING_DIM = 300  # embedding dimension
VOCAB_SIZE = 8  # vocabulary size
STD = 0.001  # standard deviation of variable initializers


class MyModel(tf.keras.Model):
    def __init__(self, adversarial=False):
        super().__init__()
        self.adversarial = adversarial

        # Embeddings variable. Using tf.Variable since tf.get_variable is not TF2 style.
        # Initialize uniformly in [-1,1].
        self.embeddings = tf.Variable(
            tf.random.uniform([VOCAB_SIZE, EMBEDDING_DIM], minval=-1.0, maxval=1.0),
            trainable=True,
            name="word_embeddings"
        )

        # GRU Cell (single layer)
        self.gru_cell = tf.keras.layers.GRUCell(MLP_HIDDEN_DIM)

        # MLP weights (instead of tf.get_variable, using tf.Variable)
        # Using dense layers with no bias and manual bias variables to match original
        self.W1 = tf.Variable(
            tf.random.normal([MLP_HIDDEN_DIM, MLP_HIDDEN_DIM], mean=0.0, stddev=STD),
            name="MLP_W1"
        )
        self.h1 = tf.Variable(
            tf.random.normal([MLP_HIDDEN_DIM], mean=0.0, stddev=STD),
            name="MLP_h1"
        )
        self.W2 = tf.Variable(
            tf.random.normal([MLP_HIDDEN_DIM, 2], mean=0.0, stddev=STD),
            name="MLP_W2"
        )
        self.h2 = tf.Variable(
            tf.random.normal([2], mean=0.0, stddev=STD),
            name="MLP_h2"
        )

    def call(self, inputs):
        # inputs: tuple of (text, text_len)
        # text: int32 tensor, shape [BATCH, MAX_LEN]
        # text_len: int32 tensor, shape [BATCH]

        text, text_len = inputs

        # Embedding lookup
        embeddings = tf.nn.embedding_lookup(self.embeddings, text)
        # embeddings shape: [BATCH, MAX_LEN, EMBEDDING_DIM]

        # Run dynamic_rnn with GRUCell manually
        # tf.keras.layers.GRUCell does not have a layered dynamic_rnn directly,
        # use tf.keras.layers.RNN with a GRUCell.
        gru_layer = tf.keras.layers.RNN(self.gru_cell, return_sequences=False, return_state=True)
        # Pack text_len for mask
        mask = tf.sequence_mask(text_len, maxlen=MAX_LEN)
        outputs, final_state = gru_layer(embeddings, mask=mask)

        # MLP forward
        # after_first_layer = relu(state * W1 + h1)
        # logits = after_first_layer * W2 + h2
        after_first_layer = tf.nn.relu(tf.matmul(final_state, self.W1) + self.h1)
        logits = tf.matmul(after_first_layer, self.W2) + self.h2

        return logits

    def loss(self, inputs, labels, emb_delta=None):
        # Compute loss optionally with embedding delta added (for adversarial perturbation)
        text, text_len = inputs
        if emb_delta is not None:
            # We add emb_delta to embeddings before lookup.
            # emb_delta shape assumed [VOCAB_SIZE, EMBEDDING_DIM]
            embeddings = self.embeddings + emb_delta
        else:
            embeddings = self.embeddings

        # Embedding lookup
        embedded_text = tf.nn.embedding_lookup(embeddings, text)

        # GRU RNN
        mask = tf.sequence_mask(text_len, maxlen=MAX_LEN)
        gru_layer = tf.keras.layers.RNN(self.gru_cell, return_sequences=False, return_state=True)
        outputs, final_state = gru_layer(embedded_text, mask=mask)

        # MLP forward
        after_first_layer = tf.nn.relu(tf.matmul(final_state, self.W1) + self.h1)
        logits = tf.matmul(after_first_layer, self.W2) + self.h2

        loss_val = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels
            )
        )
        return loss_val


def my_model_function():
    """
    Returns an instance of MyModel.
    By default, to replicate original code, adversarial mode is enabled or not depending on user.

    We create a basic model instance here with adversarial=False for simplicity,
    as the original code's adversarial logic was implemented externally.
    """
    return MyModel(adversarial=False)


def GetInput():
    """
    Returns a tuple (text, text_len) as inputs for MyModel.
    - text: random integers in [0, VOCAB_SIZE) of shape [BATCH, MAX_LEN]
    - text_len: random lengths in [1, MAX_LEN] of shape [BATCH]
    """

    import numpy as np

    np.random.seed(0)
    # Random token indices
    text = np.random.randint(0, VOCAB_SIZE, size=(BATCH, MAX_LEN), dtype=np.int32)
    # Random lengths between 1 and MAX_LEN
    text_len = np.random.randint(1, MAX_LEN + 1, size=(BATCH,), dtype=np.int32)
    return tf.constant(text), tf.constant(text_len)

