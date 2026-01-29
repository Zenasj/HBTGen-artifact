# tf.random.uniform((B, 100, 1), dtype=tf.float32) ← Input shape inferred from original code (batch size unknown, sequence length 100, 1 feature/channel)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Replicating the original LSTM model architecture from the issue:
        # Three LSTM layers with Dropout in between and a final Dense softmax layer
        self.lstm1 = layers.LSTM(256, return_sequences=True, input_shape=(100, 1))
        self.dropout1 = layers.Dropout(0.2)
        self.lstm2 = layers.LSTM(256, return_sequences=True)
        self.dropout2 = layers.Dropout(0.2)
        self.lstm3 = layers.LSTM(128)
        self.dropout3 = layers.Dropout(0.2)
        # Output dimension inferred from the number of characters in vocab (y.shape[1])
        # We will take vocab_len as 65 (rough heuristic typical for English char vocab with basic chars)
        # In practice vocab_len should be detected or set outside the model, but we embed as a param
        self.vocab_len = 65
        self.dense = layers.Dense(self.vocab_len, activation='softmax')

    def call(self, x, training=False):
        x = self.lstm1(x)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.lstm3(x)
        x = self.dropout3(x, training=training)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    # No pretrained weights are loaded here since original code trains from scratch
    return MyModel()

def GetInput():
    # Return an input tensor shaped (batch_size, seq_length, 1)
    # According to original code: seq_length=100, features=1
    # Batch size used during training was 256, but we can generate any batch size
    batch_size = 256
    seq_length = 100
    features = 1
    # The original preprocessing normalizes by vocabulary length (x = x / float(vocab_len))
    # So input should be floats in [0,1)
    input_tensor = tf.random.uniform((batch_size, seq_length, features), minval=0., maxval=1., dtype=tf.float32)
    return input_tensor

