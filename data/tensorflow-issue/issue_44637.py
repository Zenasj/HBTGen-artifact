# tf.random.uniform((B, window_length), dtype=tf.int32) ← inferred input shape based on Embedding input

import tensorflow as tf
from tensorflow import keras

# Assumptions / inferred details:
# - window_length, input_length, num_addresses, embedding_dim, NUM_HIDDEN, output_dim,
#   LEARNING_RATE, LOSS_WEIGHTS are global constants or configuration parameters.
# - Input shape for Embedding is (window_length,)
# - The original code uses SparseCategoricalCrossentropy with from_logits=True on output of shape (window_length, output_dim)
# - The model is a Bidirectional LSTM autoencoder with shared input
# - Secondary outputs and concatenation mentioned in comment are ignored here, due to missing detailed specs.
# - This model reflects the provided dummy model structure and usage.

# For completeness, we define these constants with reasonable dummy values:
window_length = 10         # Sequence length for LSTM input
input_length = 1           # Original code assumes single integer input per timestep (embedding inputs)
num_addresses = 1000       # Vocabulary size for embedding layer
embedding_dim = 64         # Dimension of embedding vectors
NUM_HIDDEN = 128           # Number of hidden units in LSTM layers
output_dim = 50            # Output dimensionality for final Dense layer per timestep
LEARNING_RATE = 0.001      # Learning rate for optimizer
LOSS_WEIGHTS = None        # No loss weights explicitly set in example; set to None
                          # Can be a dict e.g. {"the_output": 1.0} if required

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers as per the dummy code:
        self.embedding = keras.layers.Embedding(num_addresses, embedding_dim)
        self.encoder_lstm = keras.layers.Bidirectional(keras.layers.LSTM(NUM_HIDDEN))
        self.repeat_vector = keras.layers.RepeatVector(window_length)
        self.decoder_lstm = keras.layers.Bidirectional(keras.layers.LSTM(NUM_HIDDEN, return_sequences=True))
        # TimeDistributed Dense layer with ReLU activation for output
        self.time_distributed_dense = keras.layers.TimeDistributed(
            keras.layers.Dense(output_dim, activation='relu'),
            name="the_output"
        )
        
    def call(self, inputs):
        # inputs shape: (batch_size, window_length)
        x = self.embedding(inputs)                     # (batch_size, window_length, embedding_dim)
        x = self.encoder_lstm(x)                        # (batch_size, 2*NUM_HIDDEN)
        x = self.repeat_vector(x)                       # (batch_size, window_length, 2*NUM_HIDDEN)
        x = self.decoder_lstm(x)                        # (batch_size, window_length, 2*NUM_HIDDEN)
        output = self.time_distributed_dense(x)        # (batch_size, window_length, output_dim)
        return output

def my_model_function():
    # Instantiate and compile MyModel with the same optimizer and loss as dummy code
    model = MyModel()
    losses = {
        "the_output": keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    }
    # Build the model to ensure weights initialized before compiling
    # We must provide an input shape to build the model (batch size = None)
    dummy_input = tf.keras.Input(shape=(window_length,), dtype=tf.int32)
    _ = model(dummy_input)
    
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=LEARNING_RATE),
        loss=losses
    )
    return model

def GetInput():
    # Return a random input tensor matching (batch_size, window_length), integer values in [0, num_addresses)
    batch_size = 4  # Pick default batch size to simulate multi-GPU scenarios
    return tf.random.uniform(
        (batch_size, window_length), minval=0, maxval=num_addresses, dtype=tf.int32
    )

