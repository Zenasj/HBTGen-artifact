# tf.random.uniform((5, 4, 3), dtype=tf.float32) ← inferred input shape from the example tensor x in the dataset (batch=5, timesteps=4, features=3)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, rnn=None, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn

    def call(self, inputs, training=None):
        # Forward pass simply delegates to the RNN layer
        output = self.rnn(inputs, training=training)
        return output

class BarCell(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # RNNCell contract requires state_size attribute
        self.state_size = [tf.TensorShape([1])]

    def call(self, inputs, states, training=None):
        # Compute output as sum over features + 1.0 (arbitrary)
        output = tf.reduce_sum(inputs, axis=1) + tf.constant(1.0)
        # Add a loss dependent on the input (sum over features)
        self.add_loss(tf.reduce_sum(inputs))
        # Advance state by 1 (arbitrary state update logic)
        states_tplus1 = [states[0] + 1]
        return output, states_tplus1

def my_model_function():
    # Instantiate the BarCell and wrap it in an RNN layer
    cell = BarCell()
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, unroll=False)
    # Return the model instance
    return MyModel(rnn=rnn)

def GetInput():
    # Return a random tensor with shape matching the example input:
    # batch=5, timesteps=4, features=3, dtype=float32
    return tf.random.uniform((5, 4, 3), dtype=tf.float32)

