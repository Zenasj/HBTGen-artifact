# tf.random.uniform((batch_size, FLAGS.max_seq_length, hidden_size), dtype=tf.float32)
import tensorflow as tf
import numpy as np

# We will implement a fused MyModel that uses a BERT-like embedding input (simulated),
# runs bidirectional GRUs on query and document slices,
# and computes start and end logits using the AOA1 attention method.
#
# Note:
# - Actual BERT modeling, tokenization, and optimization modules are missing in the issue snippet,
#   so we simulate their core effect: an input tensor shaped [batch_size, max_seq_length, hidden_size].
# - We keep the model fully in tf.keras for modern TensorFlow 2.20.0 compatibility.
# - The orthogonal_initializer and softmax are included as given.
# - The GRU layers are built using tf.keras.layers.GRU with bidirectional wrapper, replacing deprecated tf.nn.rnn_cell.GRUCell.
# - The input shape includes batch_size, sequence length, and the embedding hidden size.
# - For simplicity, batch_size, FLAGS.max_seq_length, FLAGS.max_query_length and hidden_size are parameters.
#
# Assumptions:
# - Input shape: (batch_size, max_seq_length, hidden_size)
# - max_seq_length > max_query_length + 3, with query tokens at head of sequence followed by document tokens.
# - Input mask is a bool or float mask tensor indicating valid tokens.
#
# Outputs start_logits and end_logits tensors shaped [batch_size, doc_length]

# Parameters inferred from FLAG defaults in code snippet
FLAGS = type("FLAGS", (), {})()
FLAGS.max_seq_length = 384
FLAGS.max_query_length = 64
FLAGS.do_lower_case = True  # not used here
FLAGS.dropout_keep_prob = 1.0  # assume no dropout here
FLAGS.version_2_with_negative = False  # irrelevant for forward pass

# Orthogonal initializer as per original
def orthogonal_initializer(scale=1.1):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)
    return _initializer

def softmax(target, axis, mask, epsilon=1e-12, name=None):
    # Mask expected to be same shape as target, zeros for invalid positions
    # Implements stable masked softmax along axis
    max_axis = tf.reduce_max(target * mask + (1.0 - mask) * -1e9, axis=axis, keepdims=True)
    target_exp = tf.exp(target - max_axis) * mask
    normalize = tf.reduce_sum(target_exp, axis=axis, keepdims=True)
    softmax_out = target_exp / (normalize + epsilon)
    return softmax_out


def AOA1(q, d, batch_size, hidden_size, input_mask):
    # q: Tensor (batch_size, max_query_length+2, hidden_size/2)
    # d: Tensor (batch_size, doc_len, hidden_size/2)
    # input_mask: Tensor (batch_size, max_seq_length), 1 for valid tokens else 0

    # Shapes:
    # q: [B, Q, Hq]
    # d: [B, D, Hd] (Hd == Hq)
    # batch_size = B
    # max_query_length = Q - 2
    # doc_length = D

    max_query_len = FLAGS.max_query_length + 2
    doc_len = FLAGS.max_seq_length - max_query_len
    # compute raw similarity matrix M = d @ q^T -> shape (B, D, Q)
    M = tf.matmul(d, q, transpose_b=True)  # shape (B, D, Q)

    # Create masks for query and doc tokens from input_mask slices
    query_mask = tf.slice(input_mask, [0, 0], [batch_size, max_query_len])  # (B, Q)
    doc_mask = tf.slice(input_mask, [0, max_query_len], [batch_size, doc_len])  # (B, D)

    # Broadcast masks to create mask on M
    # M_mask[b, i, j] = doc_mask[b, i] * query_mask[b, j]
    M_mask = tf.cast(tf.expand_dims(doc_mask, -1), tf.float32) * tf.cast(tf.expand_dims(query_mask, 1), tf.float32)

    # alpha: softmax over query dimension with masking (axis=2)
    alpha = softmax(M, axis=2, mask=M_mask)

    # beta: softmax over doc dimension with masking (axis=1)
    beta = softmax(M, axis=1, mask=M_mask)

    # query importance = mean beta over doc dimension -> (B, Q)
    query_importance = tf.reduce_sum(beta, axis=1) / tf.cast(doc_len, tf.float32)  # (B, Q)

    # expand dims for matmul: (B, Q, 1)
    query_importance = tf.expand_dims(query_importance, axis=-1)

    # output s = alpha @ query_importance -> (B, D, 1)
    s = tf.matmul(alpha, query_importance)  # (B, D, 1)

    s = tf.squeeze(s, axis=-1)  # (B, D)

    return s


class MyModel(tf.keras.Model):
    def __init__(self, max_seq_length=FLAGS.max_seq_length,
                 max_query_length=FLAGS.max_query_length,
                 hidden_size=768):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.hidden_size = hidden_size

        # Document length inferred
        self.doc_length = max_seq_length - (max_query_length + 2)

        # Initialize orthogonal initializer
        self.orth_init = orthogonal_initializer()

        # Bidirectional GRUs on document and query
        # Using tf.keras.layers.Bidirectional with GRU
        # GRU hidden size matches original hidden_size from BERT
        # The original returns 2*hidden_size from concat of fwd and back
        # So the cell size here is half the output size in original code

        # Because concat of forward and backward outputs in original code makes size 2*hidden_size,
        # our hidden units per direction is hidden_size // 2
        gru_units = hidden_size // 2

        # Bidirectional GRU for document
        self.doc_bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(gru_units, return_sequences=True,
                                kernel_initializer=self.orth_init),
            merge_mode='concat', name="document_birnn")

        # Bidirectional GRU for query
        self.query_bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(gru_units, return_sequences=True,
                                kernel_initializer=self.orth_init),
            merge_mode='concat', name="query_birnn")

        # Fully connected layers on slices for start and end logits
        # Using tf.keras.layers.Dense to replace deprecated fully_connected
        dense_hidden = gru_units

        self.ques_start_dense = tf.keras.layers.Dense(dense_hidden,
                                                     activation='relu',
                                                     name='ques_start_dense')

        self.ques_end_dense = tf.keras.layers.Dense(dense_hidden,
                                                   activation='relu',
                                                   name='ques_end_dense')

        self.doc_start_dense = tf.keras.layers.Dense(dense_hidden,
                                                    activation='relu',
                                                    name='doc_start_dense')

        self.doc_end_dense = tf.keras.layers.Dense(dense_hidden,
                                                  activation='relu',
                                                  name='doc_end_dense')

    def call(self, inputs):
        # inputs is a tuple: (embedded_input, input_mask)
        # embedded_input: Tensor shape (B, max_seq_length, hidden_size)
        # input_mask: Tensor shape (B, max_seq_length), 1=valid token, 0=pad

        embedded_input, input_mask = inputs
        batch_size = tf.shape(embedded_input)[0]

        # Slice query and document portions
        query_slice = embedded_input[:, :self.max_query_length + 2, :]   # shape (B, Q+2, H)
        doc_slice = embedded_input[:, self.max_query_length + 2:, :]     # shape (B, D, H)

        # Run bidirectional rnn
        h_doc = self.doc_bi_gru(doc_slice)   # (B, D, 2*gru_units)
        h_query = self.query_bi_gru(query_slice)  # (B, Q+2, 2*gru_units)

        # Fully connected layers on query slices for start/end
        ques_start = self.ques_start_dense(h_query)  # (B, Q+2, gru_units)
        ques_end = self.ques_end_dense(h_query)      # (B, Q+2, gru_units)

        # Fully connected layers on doc slices for start/end
        doc_start = self.doc_start_dense(doc_slice)  # (B, D, gru_units)
        doc_end = self.doc_end_dense(doc_slice)      # (B, D, gru_units)

        # Compute start_logits and end_logits using AOA1 attention mechanism
        start_logits = AOA1(ques_start, doc_start, batch_size, self.hidden_size, input_mask)
        end_logits = AOA1(ques_end, doc_end, batch_size, self.hidden_size, input_mask)

        # logits shape: (B, D)
        return start_logits, end_logits


def my_model_function():
    # Return an instance of the MyModel class with default sizes
    return MyModel()


def GetInput():
    # Returns a tuple (embedded_input, input_mask)
    # embedded_input shape: (batch_size, max_seq_length, hidden_size)
    # input_mask shape: (batch_size, max_seq_length)

    batch_size = 2  # test batch size
    max_seq_length = FLAGS.max_seq_length
    max_query_length = FLAGS.max_query_length
    hidden_size = 768

    # Generate floating point embedded input tensor with random uniform values
    embedded_input = tf.random.uniform(
        shape=(batch_size, max_seq_length, hidden_size),
        dtype=tf.float32, minval=-1, maxval=1)

    # Generate input_mask: 1 for real tokens, 0 for padding
    # For simplicity, simulate all tokens valid for query + doc length
    # Here, mask all tokens up to doc length as valid, padding zero afterward (none here)
    # We use int32 mask here as float for mask in softmax (converted internally)

    # Create mask of ones for entire sequence (all valid tokens)
    input_mask = tf.ones(shape=(batch_size, max_seq_length), dtype=tf.float32)

    return (embedded_input, input_mask)

