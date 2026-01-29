# tf.random.uniform((B, input_size), dtype=tf.float32) ← inferred input shape is [batch_size, input_size]

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model reproduces an EmbeddingWrapper around an RNN cell.
    It addresses the issue that EmbeddingWrapper in older TensorFlow versions
    lacked the required @property input_size and output_size, causing
    NotImplementedError when calling the cell.

    Here, MyModel acts as the embedding wrapper which:
    - has an embedding layer
    - wraps a tf.keras.layers.SimpleRNNCell (or similar)
    - implements input_size and output_size properties
    - implements call(inputs, state) consistent with RNNCell interface
    """

    def __init__(self, vocab_size=1000, embedding_dim=32, rnn_units=64):
        """
        vocab_size: vocabulary size for the embedding input_dim
        embedding_dim: embedding vector dimension
        rnn_units: number of units in the RNN cell
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # Use a basic RNNCell - could be LSTMCell, GRUCell, etc.
        self.rnn_cell = tf.keras.layers.SimpleRNNCell(rnn_units)

    @property
    def input_size(self):
        # The input size for the cell is the vocab size before embedding
        # (since the wrapper expects input tokens)
        return self.embedding.input_dim

    @property
    def output_size(self):
        # The output_size is the RNN cell output size
        return self.rnn_cell.output_size

    @property
    def state_size(self):
        return self.rnn_cell.state_size

    def call(self, inputs, states):
        """
        Args:
            inputs: a batch of token IDs, shape [batch_size], dtype int32
            states: previous cell state tensor(s), shape depends on cell

        Returns:
            output: [batch_size, rnn_units]
            new_states: new cell state(s)
        """
        # Embed the input tokens to vectors
        embedded_inputs = self.embedding(inputs)  # Now shape [batch_size, embedding_dim]
        # Pass embedded inputs and prior states to the RNN cell
        output, new_states = self.rnn_cell(embedded_inputs, states)
        return output, new_states

def my_model_function():
    # Return an instance of MyModel with default params
    return MyModel()

def GetInput():
    # Generate a random batch of token IDs as input to MyModel
    batch_size = 4
    vocab_size = 1000  # must match model's vocab_size
    # Input is 1-D tensor of int token IDs (shape: [batch_size])
    return tf.random.uniform(shape=(batch_size,), minval=0, maxval=vocab_size, dtype=tf.int32)

