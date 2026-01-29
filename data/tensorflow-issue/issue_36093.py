# tf.random.uniform((B, 1, 1), dtype=tf.float32) ← inferred input shape from example: Input(shape=(1, 1))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, number_of_cells=2, units=10):
        super().__init__()
        # Construct multiple GRUCell instances as submodules
        # to mimic the behavior that initially caused the saving bug.
        self.cells = [tf.keras.layers.GRUCell(units, name=f"gru_cell_{i}") for i in range(number_of_cells)]
        # Wrap cells in a RNN layer
        self.rnn = tf.keras.layers.RNN(self.cells)

    def call(self, inputs, training=False):
        # inputs shape: (batch, timesteps, features)
        # Forward pass through the multi-cell RNN
        return self.rnn(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel with default 2 GRU cells, 10 units each
    return MyModel()

def GetInput():
    # Return random input matching expected input shape of MyModel
    # Based on the original example, input shape is (batch_size, 1, 1)
    # For demonstration, batch size = 4
    batch_size = 4
    # Sequence length = 1, features = 1
    return tf.random.uniform((batch_size, 1, 1), dtype=tf.float32)

