# tf.random.uniform((B, 2), dtype=tf.float32)  # input shape inferred from example inputs = tf.ones((1,2))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize a single GRUCell with dropout and recurrent dropout as in the issue example
        self.cell = tf.keras.layers.GRUCell(units=2, dropout=0.1, recurrent_dropout=0.1)

    @tf.function
    def call(self, inputs, states=None, training=True):
        """
        Run a forward pass of the GRU cell.
        This builds on the example from the issue, exposing the 'bad_infer' style usage
        inside a tf.function to illustrate the variable handling scenario.
        """
        if states is None:
            states = self.cell.get_initial_state(batch_size=tf.shape(inputs)[0], dtype=inputs.dtype)
        output, new_states = self.cell(inputs=inputs, states=states, training=training)
        return output, new_states

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return random input compatible with MyModel's expected input shape
    # The example shows inputs shape (1, 2), so batch_size=1, feature_dim=2
    return tf.random.uniform((1, 2), dtype=tf.float32)

