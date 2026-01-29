# tf.random.uniform((B, T, C), dtype=tf.float32) ← Here B=batch size, T=sequence length (timesteps), C=embedding size

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=10000, emb_size=16, rnn_size=16):
        super(MyModel, self).__init__()
        self.embedding = layers.Embedding(vocab_size, emb_size)
        # We include both versions of DynamicRNN as submodules, to reflect the original discussion
        self.dynamic_rnn_v1 = DynamicRNN(rnn_size)
        self.dynamic_rnn_v2 = DynamicRNNV2(rnn_size)

    @tf.function(jit_compile=True)
    def call(self, x, use_v2=False):
        """Run embedding and the selected RNN version, then compare outputs.
        
        Args:
          x: input tensor of shape (batch, timesteps) integer token IDs
          use_v2: boolean flag to choose which DynamicRNN to run
        
        Returns:
          A dict with keys:
            'output': The output from the chosen DynamicRNN layer, shape (batch, timesteps, rnn_size)
            'state': The final RNN state, shape (batch, rnn_size)
            'outputs_close': boolean tensor scalar if both versions are close in output numerically
            'max_diff': maximum absolute difference between outputs
        """
        emb = self.embedding(x)  # (batch, timesteps, emb_size)
        if use_v2:
            out_v2, state_v2 = self.dynamic_rnn_v2(emb)
            # Forward pass for v1 as well to compare outputs:
            out_v1, state_v1 = self.dynamic_rnn_v1(emb)
        else:
            out_v1, state_v1 = self.dynamic_rnn_v1(emb)
            out_v2, state_v2 = self.dynamic_rnn_v2(emb)

        # Compare the outputs elementwise with a tolerance
        abs_diff = tf.abs(out_v1 - out_v2)
        max_diff = tf.reduce_max(abs_diff)
        outputs_close = tf.reduce_all(abs_diff < 1e-5)

        return {
            'output': out_v1 if not use_v2 else out_v2,
            'state': state_v1 if not use_v2 else state_v2,
            'outputs_close': outputs_close,
            'max_diff': max_diff,
        }

class DynamicRNN(layers.Layer):
    """Dynamic RNN implemented with tf.function and tf.TensorArray and tf.range loop."""
    def __init__(self, rnn_size):
        super(DynamicRNN, self).__init__()
        self.rnn_size = rnn_size
        # Note: original code incorrectly used GRU layer as cell.
        # Following the original example, keep it as GRU layer with return_state=True.
        self.cell = layers.GRU(rnn_size, return_state=True)

    @tf.function
    def call(self, input_data):
        """input_data: (batch, timesteps, emb_size)"""
        outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        # Initial hidden state zero tensor
        batch_size = tf.shape(input_data)[0]
        state = tf.zeros((batch_size, self.rnn_size), dtype=tf.float32)
        seq_len = tf.shape(input_data)[1]
        for i in tf.range(seq_len):
            # Expand dims to match expected GRU input of shape (batch, 1, emb_size)
            x_i = tf.expand_dims(input_data[:, i, :], 1)
            output, state = self.cell(x_i, state)
            outputs = outputs.write(i, output)
        stacked = outputs.stack()  # (timesteps, batch, rnn_size)
        # Transpose to (batch, timesteps, rnn_size)
        return tf.transpose(stacked, [1, 0, 2]), state

class DynamicRNNV2(layers.Layer):
    """Dynamic RNN implemented with eager-style Python for loop and list concat."""
    def __init__(self, rnn_size):
        super(DynamicRNNV2, self).__init__()
        self.rnn_size = rnn_size
        self.cell = layers.GRU(rnn_size, return_state=True)

    def call(self, input_data):
        batch_size = tf.shape(input_data)[0]
        seq_len = tf.shape(input_data)[1]
        state = tf.zeros((batch_size, self.rnn_size), dtype=tf.float32)
        outputs = []
        for i in range(seq_len):
            x_i = tf.expand_dims(input_data[:, i, :], 1)
            output, state = self.cell(x_i, state)
            outputs.append(tf.expand_dims(output, 1))
        return tf.concat(outputs, axis=1), state

def my_model_function():
    # Instantiate model with sample sizes per original code
    # vocab_size and embedding size could be inputs; we fix reasonable defaults here
    vocab_size = 10000  # placeholder - in real usage, get vocab size from dataset
    emb_size = 16
    rnn_size = 16
    return MyModel(vocab_size, emb_size, rnn_size)

def GetInput():
    # Create a random integer tensor of shape (B, T) with values in vocab range matching model
    B = 32  # batch size
    T = 50  # sequence length, arbitrary chosen
    vocab_size = 10000
    # Generate integer inputs in range [0, vocab_size)
    return tf.random.uniform(shape=(B, T), minval=0, maxval=vocab_size, dtype=tf.int32)

