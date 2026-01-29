# tf.random.uniform((B, T), dtype=tf.int32) where B=30 (batch), T=16 (timesteps, sequence length)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, sequence_length, embedding_dim=32, lstm_units=50, num_layers=2, num_classes=3):
        super().__init__()
        # Embedding layer same as original keras model
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size+1,
                                                   output_dim=embedding_dim,
                                                   input_length=sequence_length,
                                                   name="embedding")
        # Build stacked LSTM cells for forward and backward directions
        self.fw_cells = [tf.keras.layers.LSTMCell(lstm_units, name=f'fw_lstm_cell_{i}') for i in range(num_layers)]
        self.bw_cells = [tf.keras.layers.LSTMCell(lstm_units, name=f'bw_lstm_cell_{i}') for i in range(num_layers)]

        # Wrap stacked cells in StackedRNNCells
        self.fw_stacked_cells = tf.keras.layers.StackedRNNCells(self.fw_cells)
        self.bw_stacked_cells = tf.keras.layers.StackedRNNCells(self.bw_cells)

        # Bidirectional RNN layer using Keras RNN wrapper and manual unrolling
        self.bidirectional_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(self.fw_stacked_cells, return_sequences=True),
            backward_layer=tf.keras.layers.RNN(self.bw_stacked_cells, return_sequences=True),
            merge_mode='concat', name='bidirectional_rnn'
        )

        # Output Dense layer per time-step (i.e. time-distributed dense)
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_classes, activation='softmax'),
            name='output_dense'
        )

    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: shape (batch_size, sequence_length), integer token IDs
        output: per time-step softmax probabilities with shape (batch_size, sequence_length, num_classes)
        """
        x = self.embedding(inputs)
        x = self.bidirectional_rnn(x, training=training)  # shape (batch_size, sequence_length, lstm_units*2)
        output = self.dense(x)  # shape (batch_size, sequence_length, num_classes)
        return output

def my_model_function():
    # Assumptions from original issue:
    # vocab size and sequence length are known from training data shapes
    # Using vocab_size=some placeholder value (e.g. 100) and sequence_length=16 (from issue)
    # These should be parameterized in real usage.
    vocab_size = 100  # inferred from vocab + 1 in original code
    sequence_length = 16
    return MyModel(vocab_size=vocab_size, sequence_length=sequence_length)

def GetInput():
    # Generate a batch of integer token sequences matching expected inputs
    # Using batch size 30 (from X.shape in issue)
    batch_size = 30
    sequence_length = 16
    # Token IDs between 0 and vocab_size (100) inclusive
    vocab_size = 100
    return tf.random.uniform((batch_size, sequence_length), minval=0, maxval=vocab_size+1, dtype=tf.int32)

