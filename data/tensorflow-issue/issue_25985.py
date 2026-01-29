# tf.random.uniform((1, 1, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Stateful LSTM layer with 1 unit, expects batch_size=1, time_steps=1, feature_dim=1
        # Note: stateful=True requires fixed batch size
        self.lstm = tf.keras.layers.LSTM(
            1,
            stateful=True,
            return_sequences=False,
            return_state=False,
        )

    def call(self, inputs):
        """
        inputs: tuple of tensors: (inputs, state_h, state_c)
        inputs:
           - inputs: shape (1, 1, 1)
           - state_h: shape (1, 1)
           - state_c: shape (1, 1)
        """
        inputs_tensor, state_h, state_c = inputs
        
        # Pass the inputs through the LSTM with initial states
        output = self.lstm(
            inputs_tensor,
            initial_state=[state_h, state_c]
        )
        return output

    def reset_states(self):
        # Custom reset_states to avoid NoneType error when using initial_state
        # Since layer is stateful, reset the states directly
        try:
            self.lstm.reset_states()
        except Exception:
            # Fallback: manually reset states if error occurs, as issue suggests
            # This is defensive, usually keras layers handle this internally
            self.lstm.states = [None, None]

def my_model_function():
    return MyModel()

def GetInput():
    # Return a tuple compatible with MyModel.call(inputs)
    # inputs tensor shape: (1, 1, 1), dtype=tf.float32 (batch=1, time steps=1, features=1)
    inputs_tensor = tf.random.uniform((1, 1, 1), dtype=tf.float32)
    # state_h and state_c shape: (1, 1) consistent with LSTM units and batch size
    state_h = tf.zeros((1, 1), dtype=tf.float32)
    state_c = tf.zeros((1, 1), dtype=tf.float32)
    return (inputs_tensor, state_h, state_c)

