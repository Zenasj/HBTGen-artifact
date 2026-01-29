# tf.random.uniform((1, 1, 1), dtype=tf.float32)  ← inferred input shape from batch_input_shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        number_of_cells = 2
        # Create two GRUCells as in the original example
        cells = [tf.keras.layers.GRUCell(10) for _ in range(number_of_cells)]
        # Stateful RNN with these cells
        self.rnn = tf.keras.layers.RNN(cells, stateful=True)
        # We add InputSpec for batch_input_shape to match original model setup
        self._set_input_spec(
            tf.keras.layers.InputSpec(batch_shape=(1, 1, 1))
        )
        
    def call(self, inputs):
        # Forward pass through the stateful RNN
        return self.rnn(inputs)
    
def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create a random tensor matching the expected input shape (batch=1, timesteps=1, features=1)
    return tf.random.uniform((1, 1, 1), dtype=tf.float32)

