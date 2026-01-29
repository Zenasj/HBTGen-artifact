# tf.random.uniform((1, 10), dtype=tf.float32), tf.random.uniform((1, 2), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters as inferred from the discussion:
        # vocab_size = 10, embedding dim = 256, lstm units = 256, input_int shape=2
        self.vocab_size = 10
        self.max_length = 10
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, 256, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(256)
        self.dense_concat = tf.keras.layers.Concatenate()
        self.dense_out = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs):
        # inputs is a tuple/list of two tensors: (input_int, input_str)
        input_int, input_str = inputs
        # input_str : shape (batch, max_length) = (1,10)
        # input_int : shape (batch, 2) = (1,2)

        x = self.embedding(input_str)   # (batch, max_length, 256)
        x = self.lstm(x)                # (batch, 256)
        z = self.dense_concat([input_int, x])  # (batch, 258)
        outputs = self.dense_out(z)     # (batch, vocab_size=10)
        return outputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a tuple of two inputs matching the model inputs:
    # - input_int of shape (1, 2), dtype float32
    # - input_str of shape (1, 10), dtype int32 (for embedding indices)
    # We use int32 for input_str because it's used as indices into embedding.
    input_int = tf.random.uniform((1, 2), minval=0, maxval=1, dtype=tf.float32)
    input_str = tf.random.uniform((1, 10), minval=1, maxval=9, dtype=tf.int32)  # vocab indices>0 since mask_zero=True
    return (input_int, input_str)

