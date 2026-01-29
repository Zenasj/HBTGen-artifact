# tf.random.uniform((1, 3, 1), dtype=tf.float32) ← Input shape from batch_shape=(1, sequence_length=3, feature_dim=1)

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Input
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use LSTM recurrent cell with 1 unit, linear activation, no bias, return sequences, no return state here.
        # Both forward and backward RNNs are stateful.
        self.sequence_length = 3
        self.feature_dim = 1
        self.units = 1
        
        # Bidirectional LSTM layers - one stateless, one stateful, same weights
        # Note: The user code initializes and transfers weights.
        self.stateless_bidi = Bidirectional(
            LSTM(self.units, activation='linear', use_bias=False, return_sequences=True,
                 return_state=False, stateful=False),
            name="stateless_bidi"
        )
        self.stateful_bidi = Bidirectional(
            LSTM(self.units, activation='linear', use_bias=False, return_sequences=True,
                 return_state=False, stateful=True),
            name="stateful_bidi"
        )
        
        # Build models for weight initialization internally
        # We will create dummy inputs to build weights:
        dummy_input = tf.zeros((1, self.sequence_length, self.feature_dim), dtype=tf.float32)
        # Build layers by calling once
        self.stateless_bidi(dummy_input)
        self.stateful_bidi(dummy_input)
        # Copy weights from stateless to stateful
        self.stateful_bidi.set_weights(self.stateless_bidi.get_weights())
        
    def call(self, inputs, initial_state=None, reset=False):
        """
        Forward pass:
        - inputs: tensor of shape (1, sequence_length=3, feature_dim=1)
        - initial_state: optional list of initial states for forward and backward LSTM in the order:
          [fwd_h, fwd_c, bwd_h, bwd_c], each (1, 1)
          (If not provided, run without initial_state.)
        - reset: Boolean indicating if stateful bidi LSTM's states should be reset before the call.
        
        Returns:
          A dict with:
            'stateless_out': output from stateless model
            'stateful_out': output from stateful model
            'equal': boolean tensor if outputs match (within tolerance)
            'diff': numeric difference tensor (stateful - stateless)
        """
        # The original issue examples used 1 batch, 3 time steps, 1 feature
        
        # Stateless output: run with initial_state if given
        if initial_state is not None:
            # Expecting 4 states for LSTM in a list: [fwd_h, fwd_c, bwd_h, bwd_c]
            # Bidirectional layer expects states as [fwd_states..., bwd_states...]
            # So we split into forward and backward initial states accordingly
            fwd_state = initial_state[:2]
            bwd_state = initial_state[2:]
            stateless_out = self.stateless_bidi(inputs, initial_state=fwd_state + bwd_state)
        else:
            stateless_out = self.stateless_bidi(inputs)
        
        if reset:
            self.stateful_bidi.reset_states()
            
        # Stateful output: need to set initial state input in the call if provided
        # Stateful RNN layers can accept initial_state argument on call
        if initial_state is not None:
            fwd_state = initial_state[:2]
            bwd_state = initial_state[2:]
            stateful_out = self.stateful_bidi(inputs, initial_state=fwd_state + bwd_state)
        else:
            stateful_out = self.stateful_bidi(inputs)
        
        # Compute difference and comparison
        diff = stateful_out - stateless_out
        # Use a small tolerance comparison
        equal = tf.reduce_all(tf.abs(diff) < 1e-5)
        
        return {
            'stateless_out': stateless_out,
            'stateful_out': stateful_out,
            'equal': equal,
            'diff': diff,
        }

def my_model_function():
    # Return instance of MyModel
    return MyModel()

def GetInput():
    # Return a sample input tensor of shape (1, 3, 1) matching the model input.
    # Based on the example code, input could be normal random or fixed.
    # Use uniform random inputs to avoid randomness in examples:
    return tf.random.uniform((1, 3, 1), dtype=tf.float32)

