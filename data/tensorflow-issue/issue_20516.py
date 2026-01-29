# tf.random.uniform((1, 1), dtype=tf.float32) ← inferred input shape used for the example call to BasicLSTMCell

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The example in the issue uses BasicLSTMCell repeatedly assigned,
        # which causes checkpoint restoration problems due to variable key mismatches.
        # We simulate the same structure and the problem scenario.
        self.cell = tf.keras.layers.LSTMCell(4)  # Equivalent to tf.nn.rnn_cell.BasicLSTMCell(4)
        # Intentionally assign again to simulate double assignment (commented as this breaks checkpointing)
        # self.cell = tf.keras.layers.LSTMCell(4)

        # Create an Adam optimizer - this is saved as part of the checkpoint as well
        self.optimizer = tf.keras.optimizers.Adam()

        # Create trainable variables to have something checkpointable
        # We use the LSTMCell's variables inside the layer already
        # Example dense layer to simulate another variable, like in the issue
        self.dense = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs, states=None, training=False):
        if states is None:
            batch_size = tf.shape(inputs)[0]
            states = self.cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=tf.float32)
        output, new_states = self.cell(inputs, states=states, training=training)
        output = self.dense(output)
        return output, new_states

def my_model_function():
    # Return initialized MyModel instance
    return MyModel()

def GetInput():
    # Return a random tensor input matching the LSTMCell input shape
    # Based on example in the issue: input to BasicLSTMCell is shape (batch=1, input_dim=1)
    # Here we create (batch=1, time=1, features=1) and squeeze dims for cell input (1,1)
    # Since call expects (batch, features), we generate (1,1)
    return tf.random.uniform((1, 1), dtype=tf.float32)

