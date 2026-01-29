# tf.random.uniform((B, None), dtype=tf.int32) ← Input shape inferred as (batch_size, sequence_length)
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        max_features = 1_000_000  # Vocabulary size from issue
        embedding_dim = 128
        lstm_units = 64
        # Embedding layer: input_dim=max_features, output_dim=128
        self.embedding = layers.Embedding(max_features, embedding_dim)
        # Two Bidirectional LSTM layers. First returns sequences, second returns last output
        self.bilstm1 = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))
        self.bilstm2 = layers.Bidirectional(layers.LSTM(lstm_units))
        # Final Dense layer with sigmoid for binary classification
        self.classifier = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        output = self.classifier(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input of shape (batch_size=32, sequence_length=200)
    # Note: sequences are integer token IDs from 1 to max_features (approx 1,000,000)
    batch_size = 32
    sequence_length = 200  # maxlen in the provided code
    max_features = 1_000_000
    # Generate integer token indices from 1 to max_features-1 (0 typically reserved for padding)
    # Using dtype int32 as model expects int32 input shape=(None,)
    return tf.random.uniform(
        shape=(batch_size, sequence_length),
        minval=1, maxval=max_features,
        dtype=tf.int32
    )

