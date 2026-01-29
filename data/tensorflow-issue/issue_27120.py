# tf.random.uniform((batch_size, sequence_length, feature_dim), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, hidden_dim=128, en_vocab_size=10000, vi_vocab_size=10000, loss_decay=1.0):
        super(MyModel, self).__init__()
        # Word embeddings
        self.en_word_embedding = tf.keras.layers.Embedding(en_vocab_size + 1, 300, mask_zero=True)
        self.vi_word_embedding = tf.keras.layers.Embedding(vi_vocab_size + 1, 300, mask_zero=True)
        # Encoder LSTM
        self.enc_lstm = tf.keras.layers.LSTM(hidden_dim, return_state=True)
        # Decoder LSTM
        self.dec_lstm = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)
        # Output Dense layer with TimeDistributed wrapper
        self.output_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vi_vocab_size + 1, use_bias=False))
        self.loss_decay = loss_decay
        self.target_vocab_size = vi_vocab_size + 1

    def encode(self, seq):
        # seq: (batch, time) integer token ids input
        enc_output, enc_h, enc_c = self.enc_lstm(seq)
        return enc_h, enc_c

    def decode(self, seq, state=None):
        # seq: (batch, time) tokens for decoder input
        dec_output, dec_h, dec_c = self.dec_lstm(seq, initial_state=state)
        return dec_output, (dec_h, dec_c)

    def call(self, x1, x2):
        # x1: source sequence tensor (batch, time)
        # x2: target sequence tensor (batch, time)
        # Use CPU for embedding (from original code sample)
        with tf.device("/CPU:0"):
            x1_emb = self.en_word_embedding(x1)
            x2_emb = self.vi_word_embedding(x2)
        enc_state = self.encode(x1_emb)
        dec_output, _ = self.decode(x2_emb, enc_state)
        y = self.output_dense(dec_output)
        return y

    @tf.function
    def loss(self, y_pred, y_true):
        # Compute cross-entropy loss with flattening for sparse targets,
        # scaled by loss_decay factor
        logits = tf.reshape(y_pred, shape=(-1, self.target_vocab_size))
        labels = tf.reshape(y_true, shape=(-1,))
        loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss_val = tf.reduce_sum(loss_val) / self.loss_decay
        return loss_val


def my_model_function():
    # Initialize the model with some default vocab sizes and loss decay
    # We use 10,000 as vocab sizes just as a reasonable default assumption
    model = MyModel(hidden_dim=128, en_vocab_size=10000, vi_vocab_size=10000, loss_decay=1000.0)
    return model


def GetInput():
    # The model expects two input tensors: x1 and x2 representing integer token sequences.
    # Assume batch size 4, sequence length 20, dtype int32
    batch_size = 4
    seq_len = 20
    # Using vocab indices from 1 to 10000 (0 is padding)
    x1 = tf.random.uniform((batch_size, seq_len), minval=1, maxval=10000, dtype=tf.int32)
    x2 = tf.random.uniform((batch_size, seq_len), minval=1, maxval=10000, dtype=tf.int32)
    return (x1, x2)

