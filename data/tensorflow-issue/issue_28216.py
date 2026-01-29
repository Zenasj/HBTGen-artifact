# tf.random.normal((20, 50), dtype=tf.float32) ← Assumed input shape from example usage: batch_size=20, embedding_size=50

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_layers=2, num_units=128):
        super().__init__()
        
        # Create LSTMCells for stacking
        self.num_layers = num_layers
        self.num_units = num_units
        
        self.cells = [
            tf.keras.layers.LSTMCell(num_units)
            for _ in range(num_layers)
        ]
        
        # StackedRNNCells represents MultiRNNCell from tf1.x
        self.cell = tf.keras.layers.StackedRNNCells(self.cells)
    
    def call(self, x, state):
        # The old TF1.x MultiRNNCell used a concatenated state vector for all layers.
        # Here, state is expected as a concatenation of all LSTM states (c and h for each layer).
        # We split concatenated state into 4 splits because:
        # Each LSTMCell state: (c, h), each of shape [batch_size, num_units=128]
        # For num_layers=2, total state splits = 4 (c0, h0, c1, h1)
        
        # Split the concatenated state (shape: [batch, 512]) into 4 tensors of shape [batch, 128]
        split_state = tf.split(state, [self.num_units]*4, axis=1)
        
        # Restructure into tuples: ((c0, h0), (c1, h1)) expected by StackedRNNCells
        tupled_state = ((split_state[0], split_state[1]), (split_state[2], split_state[3]))
        
        # Run one step of the stacked RNN cell
        output, new_state = self.cell(x, tupled_state)
        
        # new_state is ((c0', h0'), (c1', h1')), concat back to single tensor shape [batch, 512]
        # Concatenate in order: c0', h0', c1', h1'
        concat_new_state = tf.concat([
            new_state[0][0], new_state[0][1], new_state[1][0], new_state[1][1]
        ], axis=1)
        
        return output, concat_new_state


def my_model_function():
    """
    Returns an instance of MyModel initialized with default parameters.
    """
    return MyModel()


def GetInput():
    """
    Returns a tuple (x, state) matching the input expected by MyModel.call:
    - x: a float32 tensor of shape [batch_size, embedding_size], as example 20x50
    - state: a float32 concatenated tensor of all LSTM states,
      shape [batch_size, num_layers * 2 * num_units] = [20, 512]
    """
    batch_size = 20
    embedding_size = 50
    num_layers = 2
    num_units = 128
    
    # Random input x
    x = tf.random.normal([batch_size, embedding_size], dtype=tf.float32)
    
    # For initial state, MyModel expects a concatenated tensor equivalent to
    # the concatenation of all (c, h) states for each LSTM layer.
    # We'll create initial zeros and concatenate for shape [batch, num_layers * 2 * num_units]
    state = tf.zeros([batch_size, num_layers * 2 * num_units], dtype=tf.float32)
    
    return x, state

