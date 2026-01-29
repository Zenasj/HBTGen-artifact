# tf.random.uniform((BATCH_SIZE, SEQ_LENGTH, INPUT_DIM), dtype=tf.float32)

import tensorflow as tf
import numpy as np

# Assumptions and inferred from issue:
# - Input shape: (batch_size, 24000, 598)
# - Output shape: (batch_size, 24000, 3)
# - 3 LSTM layers with 100 units each, BatchNormalization after each
# - Dense layer with softmax activation producing output per timestep
# - Single model class encapsulating the architecture with options for CuDNNLSTM or regular LSTM
# - Supports dynamic batch size and sequence length
# - Compatible with TF 2.20.0 and tf.function jit_compile=True
# - Dropout/recurrent dropout used only in non-CuDNN LSTM case (CuDNNLSTM does not support these params)
# - Input dtype tf.float32
# - Output is the model's forward pass logits (softmax probabilities per timestep)

# To keep things self-contained, no Estimator code here,
# but model replicates the core architecture described.
# Usage with Estimator mixing caused OOM issues due to large sequence length,
# but here we just implement the model itself as requested.

USE_CUDNN_LSTM = False  # Toggle to True to use CuDNNLSTM if GPU & environment available

BATCH_SIZE = 1          # Example config from issue when using estimator path
SEQ_LENGTH = 24000      # Max sequence length tested in issue
INPUT_DIM = 598
OUTPUT_DIM = 3

NP_DTYPE = np.float32
TF_DTYPE = tf.float32


class MyModel(tf.keras.Model):
    def __init__(self,
                 batch_size=BATCH_SIZE,
                 seq_length=SEQ_LENGTH,
                 input_dim=INPUT_DIM,
                 output_dim=OUTPUT_DIM,
                 use_cudnn=USE_CUDNN_LSTM):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_cudnn = use_cudnn

        self.n_layers = 3
        self.lstm_units = [100, 100, 100]

        self.dropout = 0.1
        self.recurrent_dropout = 0.1

        # Build LSTM + BatchNorm layers list
        self.lstm_layers = []
        self.bn_layers = []

        for i in range(self.n_layers):
            if self.use_cudnn:
                # CuDNNLSTM constructor in TF2 does not support dropout params; also batch_input_shape not needed explicitly here.
                self.lstm_layers.append(
                    tf.keras.layers.CuDNNLSTM(
                        self.lstm_units[i],
                        return_sequences=True,
                        stateful=False,
                        name=f'cudnn_lstm_{i+1}'))
            else:
                self.lstm_layers.append(
                    tf.keras.layers.LSTM(
                        self.lstm_units[i],
                        return_sequences=True,
                        stateful=False,
                        activation='tanh',
                        kernel_initializer='he_normal',
                        dropout=self.dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        name=f'lstm_{i+1}'))
            self.bn_layers.append(
                tf.keras.layers.BatchNormalization(
                    momentum=0.99,
                    epsilon=0.001,
                    center=True,
                    scale=True,
                    beta_initializer='zeros',
                    gamma_initializer='ones',
                    moving_mean_initializer='zeros',
                    moving_variance_initializer='ones',
                    name=f'batch_norm_{i+1}'))

        # Final Dense layer across time steps outputting class probabilities
        self.dense = tf.keras.layers.Dense(
            self.output_dim,
            activation='softmax',
            kernel_initializer='he_normal',
            name='dense_output')

        # Optimizer and loss defined externally, so omitted in model itself


    def call(self, inputs, training=False):
        """
        Forward pass. Inputs shape: (batch_size, seq_length, input_dim)
        Outputs shape: (batch_size, seq_length, output_dim) with softmax probabilities
        """
        x = inputs
        for lstm_layer, bn_layer in zip(self.lstm_layers, self.bn_layers):
            x = lstm_layer(x, training=training)
            x = bn_layer(x, training=training)
        x = self.dense(x)
        return x


def my_model_function():
    """
    Instantiate MyModel with default parameters as given in the issue.
    """
    return MyModel(batch_size=BATCH_SIZE,
                   seq_length=SEQ_LENGTH,
                   input_dim=INPUT_DIM,
                   output_dim=OUTPUT_DIM,
                   use_cudnn=USE_CUDNN_LSTM)


def GetInput():
    """
    Return a random float32 tensor with shape (BATCH_SIZE, SEQ_LENGTH, INPUT_DIM)
    matching the model input specification.
    Values uniform in [0,1).
    """
    return tf.random.uniform(
        shape=(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM),
        dtype=TF_DTYPE)


# Notes:
# - This model is compatible with @tf.function(jit_compile=True) for XLA.
# - Dropout only applies when training=True.
# - CuDNNLSTM usage should only be enabled if environment supports it;
#   otherwise fallback to LSTM as in the original issue code.
# - Input tensor shape and dtypes matched exactly from the provided code.

