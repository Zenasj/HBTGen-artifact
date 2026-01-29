# tf.random.uniform((1, None), dtype=tf.int32) ← Input shape is (batch=1, time steps=None) with integer tokens

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Activation, GRU, LSTM, SimpleRNN, Input
from tensorflow.keras.utils import to_categorical
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, rnn_type='GRU', hidden_size=50, max_output=10):
        """
        Initialize the model with specified RNN cell type ('GRU', 'LSTM', 'SimpleRNN'),
        hidden layer size, and maximum output vocabulary size.
        
        The model uses stateful RNN cells with return_sequences=True,
        an embedding layer initialized to identity matrix, a Dense projection,
        and softmax activation to produce categorical predictions.
        """
        super().__init__()
        rnn_map = {'GRU': GRU, 'LSTM': LSTM, 'SimpleRNN': SimpleRNN}
        if rnn_type not in rnn_map:
            raise ValueError(f"Invalid rnn_type {rnn_type}, choose from {list(rnn_map.keys())}")

        self.max_output = max_output
        self.hidden_size = hidden_size
        
        self.embedding = Embedding(
            input_dim=self.max_output,
            output_dim=self.max_output,
            embeddings_initializer=tf.keras.initializers.Identity(),
            trainable=True)
        
        # Stateful RNN cell for online sequence processing with unknown time steps
        self.rnn = rnn_map[rnn_type](hidden_size, return_sequences=True, stateful=True)
        
        self.dense = Dense(self.max_output)
        self.activation = Activation('softmax')

    def call(self, inputs):
        """
        Forward pass taking input tensor of shape (batch=1, time_steps,).
        Outputs probability distribution over classes for every time step.
        """
        x = self.embedding(inputs)  # (batch, time_steps, embedding_dim=max_output)
        x = self.rnn(x)             # (batch, time_steps, hidden_size)
        x = self.dense(x)           # (batch, time_steps, max_output)
        return self.activation(x)   # (batch, time_steps, max_output)

def my_model_function():
    """
    Return an instance of MyModel configured to mimic the example:
    - RNN type 'GRU' (as in the showcased issue)
    - Hidden size 50
    - max_output classes 10
    """
    return MyModel(rnn_type='GRU', hidden_size=50, max_output=10)

def GetInput():
    """
    Returns a sample input tensor matching the expected input shape and dtype for MyModel.
    The input shape is (batch=1, sequence_length), sequence_length is 10 as example.
    Values are integers in [0, max_output - 1].
    """
    max_output = 10
    sequence_length = 10
    # Use numpy array with integers representing token indices
    sample_sequence = np.array([[1,3,2,4,5,3,2,3,4,5]], dtype=np.int32)
    # Convert to tf.Tensor
    return tf.convert_to_tensor(sample_sequence, dtype=tf.int32)

