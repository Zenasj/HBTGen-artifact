# tf.random.uniform((B, 30, 1), dtype=tf.float32) ← inferred input shape: batch size variable, sequence length 30, feature dim 1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the original LSTM model from the issue
        # Layers: 
        # LSTM 258 units (return sequences=True)
        # LSTM 124 units (return sequences=True, dropout=0.2, recurrent_dropout=0.2)
        # LSTM 32 units (no return sequences, dropout=0.2, recurrent_dropout=0.2)
        # Dense(1, activation='sigmoid')
        
        # Note: recurrent_dropout is not supported in GPU/XLA and in some TF versions; 
        # but we keep it here for fidelity. It may be disabled or replaced if needed.
        
        self.lstm1 = tf.keras.layers.LSTM(
            258, return_sequences=True, input_shape=(None,1))
        self.lstm2 = tf.keras.layers.LSTM(
            124, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        self.lstm3 = tf.keras.layers.LSTM(
            32, dropout=0.2, recurrent_dropout=0.2)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # Ensure inputs are float32 (as the dtype mismatch is the root cause of the issue):
        if inputs.dtype != tf.float32:
            inputs = tf.cast(inputs, tf.float32)
        x = self.lstm1(inputs, training=training)
        x = self.lstm2(x, training=training)
        x = self.lstm3(x, training=training)
        out = self.dense(x)
        return out

def my_model_function():
    # Return an instance of MyModel 
    # Normally, you might load weights here if available,
    # or train separately. Here, just instantiate fresh model.
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape:
    # Batch size = 256 (typical), sequence length = 30, feature dim = 1
    # float32 dtype as required by the model
    # This input shape is inferred from the TimeseriesGenerator and model input shape.
    B = 256
    seq_len = 30
    feature_dim = 1
    return tf.random.uniform((B, seq_len, feature_dim), dtype=tf.float32)

