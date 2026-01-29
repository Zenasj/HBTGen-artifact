# tf.random.uniform((B, T, 40), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Combined model demonstrating the original RNNTModel with two versions
    of recurrent_activation usage in LSTM:
    - One using recurrent_activation='sigmoid'  (enables CuDNN kernel)
    - One using recurrent_activation=tf.keras.activations.sigmoid (no CuDNN)
    
    The forward pass runs both variants on identical inputs and compares outputs.
    This comparison highlights the difference discussed in the issue:
    using string 'sigmoid' triggers CuDNN kernel and convergence,
    using tf.keras.activations.sigmoid triggers slower standard LSTM without convergence.
    
    Output is a dictionary containing outputs from both variants and a boolean tensor 
    indicating if they are close within a tolerance.
    
    Assumptions:
    - Input shape: [batch, timesteps, 40] feature vectors.
    - The model includes simple sequence length input as int tensor.
    - For this demonstration, labels and loss are not computed.
    """

    def __init__(self, num_lstm_layers=2, lstm_cell_size=32, dropout=0.0, vocab_size=29):
        super().__init__()
        self.num_lstm_layers = num_lstm_layers
        self.lstm_cell_size = lstm_cell_size
        self.dropout = dropout
        self.vocab_size = vocab_size

        # Pooling layer after first LSTM layer, same for both variants
        self.pooling = tf.keras.layers.MaxPool1D(pool_size=3, padding="same")

        # Build two parallel transcription networks:
        # One with recurrent_activation='sigmoid' (string)
        self.trans_layers_str = []
        recurrent_activation_str = 'sigmoid'
        for _ in range(num_lstm_layers):
            self.trans_layers_str.append(
                tf.keras.layers.LSTM(
                    self.lstm_cell_size,
                    return_sequences=True,
                    dropout=self.dropout,
                    recurrent_activation=recurrent_activation_str))

        # One with recurrent_activation=tf.keras.activations.sigmoid (callable)
        self.trans_layers_callable = []
        recurrent_activation_callable = tf.keras.activations.sigmoid
        for _ in range(num_lstm_layers):
            self.trans_layers_callable.append(
                tf.keras.layers.LSTM(
                    self.lstm_cell_size,
                    return_sequences=True,
                    dropout=self.dropout,
                    recurrent_activation=recurrent_activation_callable))

        # Shared dense layer for both outputs to simulate CTC output layer
        self.ctc_layer = tf.keras.layers.Dense(1 + self.vocab_size, name='ctc')

    def _run_transcription(self, x, x_len, trans_layers):
        seq_len = x_len
        output = tf.clip_by_value(x, -3.0, 3.0)
        for l in range(self.num_lstm_layers):
            mask = tf.sequence_mask(seq_len)
            output = trans_layers[l](output, mask=mask, training=True)
            if l == 0:
                seq_len = tf.cast(tf.math.ceil(seq_len / 3), dtype=tf.int32)
                output = self.pooling(output, training=True)
        mask = tf.sequence_mask(seq_len)
        output = trans_layers[-1](output, mask=mask, training=True)
        return output, seq_len

    def call(self, inputs, training=True):
        """
        Expects input tuple (x, x_len)
        x: float32 tensor of shape [batch, timestep, 40]
        x_len: int32 tensor of shape [batch] - sequence lengths
        """
        x, x_len = inputs

        # Run transcription network with string recurrent_activation (cuDNN path)
        trans_out_str, out_len_str = self._run_transcription(x, x_len, self.trans_layers_str)
        ctc_out_str = self.ctc_layer(trans_out_str, training=training)

        # Run transcription network with callable recurrent_activation (standard LSTM path)
        trans_out_callable, out_len_callable = self._run_transcription(x, x_len, self.trans_layers_callable)
        ctc_out_callable = self.ctc_layer(trans_out_callable, training=training)

        # Compare outputs numerically, within a tolerance
        # Outputs shapes: [batch, timestep', vocab_size+1]
        # We compare ctc outputs (logits) for closeness.
        are_close = tf.reduce_all(
            tf.abs(ctc_out_str - ctc_out_callable) < 1e-5,
            axis=[1, 2])  # shape: [batch]

        return {
            'ctc_out_str': ctc_out_str,
            'output_len_str': out_len_str,
            'ctc_out_callable': ctc_out_callable,
            'output_len_callable': out_len_callable,
            'outputs_close': are_close
        }

def my_model_function():
    """
    Return an instance of MyModel with default parameters reflecting
    the example from the issue:
    - 2 LSTM layers
    - 32 units each
    - vocab_size=29 (from example)
    - dropout=0.0
    """
    return MyModel(num_lstm_layers=2, lstm_cell_size=32, dropout=0.0, vocab_size=29)

def GetInput():
    """
    Returns a random input tuple (x, x_len) compatible with MyModel.
    
    - x: tf.float32 tensor with shape = [batch, timestep, 40], features clipped between -3 and 3 by model.
    - x_len: tf.int32 tensor shape [batch], sequence lengths <= timestep.
    
    Using batch=4, timestep=100 as example.
    """
    batch = 4
    timestep = 100
    feature_dim = 40

    x = tf.random.uniform(shape=[batch, timestep, feature_dim], minval=-5.0, maxval=5.0, dtype=tf.float32)
    # Provide sequence lengths between 50 and 100 (inclusive)
    x_len = tf.random.uniform(shape=[batch], minval=50, maxval=timestep+1, dtype=tf.int32)
    return (x, x_len)

