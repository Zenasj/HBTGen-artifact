# tf.random.uniform((B, T, 2), dtype=tf.float32) ← Input shape inferred as (batch_size, time_steps, features=2) from example inputs

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a custom RNN cell with high-order state_size (nested tuple of TensorShapes)
        class NewCell(layers.Layer):
            def __init__(self, name=None):
                super().__init__(name=name)
                # The state_size is a tuple of:
                #   - a TensorShape [1, 2]
                #   - a tuple of two TensorShapes: [1, 2] and [2, 3]
                # This structure is more complex than usual single TensorShape or single integer.
                self.state_size = (
                    tf.TensorShape([1, 2]),
                    (tf.TensorShape([1, 2]), tf.TensorShape([2, 3]))
                )
                # output_size is a tuple with one TensorShape of [1, 2]
                self.output_size = (tf.TensorShape([1, 2]),)

            def call(self, inputs, states):
                # For demonstration, simply return inputs as output and states unchanged
                # The real logic can be implemented by user
                # This replicates the "print 'everything is safe and sound!'" behavior but without print for TF graph compatibility
                return inputs, states
        
        self.rnn_layer = layers.RNN(NewCell())

    def call(self, inputs):
        # Forward inputs through the RNN layer that uses NewCell
        return self.rnn_layer(inputs)


def my_model_function():
    # Return an instance of MyModel with the custom RNN cell that supports complex high-order state_size
    return MyModel()

def GetInput():
    # Return a random tensor input matching the input expected by MyModel
    # From example: shape (2, 1, 2) corresponds to batch=2, time=1, features=2
    return tf.random.uniform((2, 1, 2), dtype=tf.float32)

