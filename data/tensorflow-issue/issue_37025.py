# tf.random.uniform((1, 5, 64), dtype=tf.float32) ← Assumed input shape: batch=1, time=5, features=64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct multi-layer LSTM cell stack
        # Using 3 layers of TFLiteLSTMCell with 192 units each, as per original code.
        # Since tensorflow_core and TFLiteLSTMCell are not standard imports,
        # we use tf.keras.layers.LSTMCell as placeholder for demonstration.
        # In practice, replace with TFLiteLSTMCell from experimental examples if available.
        self.num_layers = 3
        self.units = 192
        
        # Create list of LSTMCells
        self.cells = [tf.keras.layers.LSTMCell(self.units, name=f'lstm{i}') for i in range(self.num_layers)]
        self.rnn_layer = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(self.cells), 
                                             return_state=True, return_sequences=True, name='stacked_lstm')
        # Final dense layer with sigmoid activation
        self.dense = tf.keras.layers.Dense(64, activation='sigmoid', name='fin_dense')

        # Initialize state variables to make the model stateful
        # We create trainable=False variables to hold cell states (c) and hidden states (h) for each layer
        self.state_c_vars = [tf.Variable(tf.zeros([1, self.units]), trainable=False, name=f'state_c_{i}') 
                             for i in range(self.num_layers)]
        self.state_h_vars = [tf.Variable(tf.zeros([1, self.units]), trainable=False, name=f'state_h_{i}') 
                             for i in range(self.num_layers)]

    def call(self, inputs, training=None):
        # inputs : Tensor of shape [batch, time, features]
        # Compose initial states for the multi-layer LSTM from the stored state variables
        initial_states = []
        for c_var, h_var in zip(self.state_c_vars, self.state_h_vars):
            initial_states.append([c_var, h_var])
        # Flatten initial_states list as layers expect [c0, h0, c1, h1, ...]
        # Keras LSTMCell expects initial_state as [h, c] per cell, but we keep convention [c, h]
        # So we swap order if necessary.
        # Keras LSTMCell signature expects [hidden_state, cell_state], so swap per cell.
        # Our stored vars are (c, h), so swap to (h, c)
        k_initial_states = []
        for c_var, h_var in zip(self.state_c_vars, self.state_h_vars):
            k_initial_states.append(h_var)
            k_initial_states.append(c_var)

        # Run the stacked LSTM with initial states
        # rnn_layer returns (all_outputs, *last_states)
        outputs_and_states = self.rnn_layer(inputs, initial_state=k_initial_states, training=training)
        outputs = outputs_and_states[0]  # sequence output [batch, time, units]
        last_states = outputs_and_states[1:]  # states: list with len=num_layers*2 (h, c) per layer

        # Extract last output in time dimension (assuming return_sequences=True)
        last_output = outputs[:, -1, :]  # [batch, units]

        # Update stored state variables with new states
        # last_states returned in order: h0, c0, h1, c1, ...
        # So we map these back to state_c_vars and state_h_vars appropriately
        for i in range(self.num_layers):
            # last_states has h,i at 2*i, c,i at 2*i+1
            h_new = last_states[2*i]
            c_new = last_states[2*i+1]
            self.state_h_vars[i].assign(h_new)
            self.state_c_vars[i].assign(c_new)

        # Pass last output through Dense layer with sigmoid activation
        out = self.dense(last_output)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor with shape [1, 5, 64]
    # batch=1, time=5, input features=64, matching original input shape in the code
    return tf.random.uniform((1, 5, 64), dtype=tf.float32)

