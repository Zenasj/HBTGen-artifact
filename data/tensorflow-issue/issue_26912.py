# tf.random.uniform((11, 1771), dtype=tf.float32) ← Model expects 11 separate inputs each with shape (1771,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # First submodel: simulation of the loaded 'first_model'
        # Input shape: (1771,), output shape: (161,)
        # We'll create a simple fully connected network as a placeholder for 'first_model'.
        # Since original model details are missing, assume 2 dense layers.
        self.first_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(1771,)),
            tf.keras.layers.Dense(161)
        ], name='first_model')

        # Second submodel: simulation of the loaded 'second_model'
        # Input shape: concatenation of 11 times 161 = 1771
        # Output shape: (161,)
        # Again a simple dense network as placeholder.
        self.second_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(161 * 11,)),
            tf.keras.layers.Dense(161)
        ], name='second_model')

        # Concatenate layer doesn't need to be explicitly defined since can use tf.concat

    def call(self, inputs):
        # inputs: list or tuple of length 11, each tensor shape (batch_size, 1771)
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 11:
            raise ValueError("Expected input to be a list/tuple of 11 tensors each of shape (batch_size, 1771)")

        # Pass each input through first_model independently
        first_outputs = [self.first_model(x) for x in inputs]  # each output shape (batch_size, 161)

        # Concatenate outputs along last dimension: shape (batch_size, 161 * 11)
        concatenated = tf.concat(first_outputs, axis=-1)

        # Pass concatenated vector through second_model
        output = self.second_model(concatenated)  # shape (batch_size, 161)

        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a list of 11 tensors, each of shape (batch_size, 1771)
    # Pick batch_size = 4 (arbitrary small batch for demo)
    batch_size = 4
    # Generate random float32 tensors matching inputs expected by first_model inputs
    inputs = [tf.random.uniform((batch_size, 1771), dtype=tf.float32) for _ in range(11)]
    return inputs

