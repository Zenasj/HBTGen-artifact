# tf.random.uniform((B, 5, 5), dtype=tf.float32) ← Inferred input shape from original code: (batch_size, lag=5, num_parameters=5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Submodel 1: LSTM-based model
        self.lstm_layer = tf.keras.layers.LSTM(5, return_sequences=False)
        self.dense_lstm = tf.keras.layers.Dense(1, activation="linear")
        
        # Submodel 2: GRU-based model
        self.gru_layer = tf.keras.layers.GRU(5, return_sequences=False)
        self.dense_gru = tf.keras.layers.Dense(1, activation="linear")

    def call(self, inputs, training=False):
        """
        Forward pass computes outputs from both LSTM and GRU submodels,
        then compares their outputs approximately for reproducibility analysis.
        Outputs a dictionary with both predictions and a bool tensor indicating
        whether they are close within a tolerance.
        """
        lstm_out = self.lstm_layer(inputs, training=training)
        lstm_pred = self.dense_lstm(lstm_out)
        
        gru_out = self.gru_layer(inputs, training=training)
        gru_pred = self.dense_gru(gru_out)
        
        # Compare results: float32 default tolerance like np.allclose default
        close = tf.math.abs(lstm_pred - gru_pred) < 1e-6
        
        # Return dictionary for clarity:
        # 'lstm_pred': LSTM model output prediction
        # 'gru_pred': GRU model output prediction
        # 'outputs_close': boolean tensor indicating closeness of results
        return {
            'lstm_pred': lstm_pred,
            'gru_pred': gru_pred,
            'outputs_close': close
        }

def my_model_function():
    # Return an instance of MyModel
    # This model fuses both LSTM and GRU to illustrate the behavior comparison described
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected shape of MyModel input.
    # From the original issue: input_shape=(lag=5, num_parameters=5)
    # Batch size is dynamic; choose a reasonable batch size, e.g., 4.
    batch_size = 4
    lag = 5
    num_parameters = 5
    # Use float32 by default as typical for TF models
    return tf.random.uniform((batch_size, lag, num_parameters), dtype=tf.float32)

