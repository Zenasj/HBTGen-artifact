# tf.random.uniform((BATCH_SIZE, SEQ_LEN), dtype=tf.int64)
import tensorflow as tf

VOCAB_SIZE = 32000
BATCH_SIZE = 20
SEQ_LEN = 20  # inferred from dataset shape and usage

class MyModel(tf.keras.Model):
    def __init__(self, target_vocab_size):
        super(MyModel, self).__init__()
        self.d_model = 1024
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.d_model)
        self.body = tf.keras.layers.Dense(self.d_model)

    def call(self, inputs, training=False):
        # inputs shape: (batch, seq_len)
        x = self.embedding(inputs)  # shape: (batch, seq_len, d_model)
        x = self.body(x)            # shape: (batch, seq_len, d_model)
        # logits shape: (batch, seq_len, vocab_size)
        logits = tf.matmul(x, self.embedding.embeddings, transpose_b=True)
        return logits


def my_model_function():
    # Return an instance of MyModel
    return MyModel(VOCAB_SIZE)


def GetInput():
    # Returns a random tensor input compatible with MyModel: shape (BATCH_SIZE, SEQ_LEN), dtype int64
    # Here, seq_len fixed to 20 as per original dataset
    return tf.random.uniform(
        (BATCH_SIZE, SEQ_LEN), minval=0, maxval=VOCAB_SIZE, dtype=tf.int64
    )

