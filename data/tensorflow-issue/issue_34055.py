# tf.random.uniform((1, 3, 1), dtype=tf.float32) ← Input shape: batch=1, timesteps=3, features=1

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Bidirectional SimpleRNN: single unit, no bias, linear activation,
        # return_sequences=True, return_state=False, initially stateful=False
        self.stateless_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.SimpleRNN(
                units=1,
                activation=None,
                use_bias=False,
                return_sequences=True,
                return_state=False,
                stateful=False,
            )
        )
        # Stateful bidirectional RNN with same config but stateful=True
        self.stateful_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.SimpleRNN(
                units=1,
                activation=None,
                use_bias=False,
                return_sequences=True,
                return_state=False,
                stateful=True,
            )
        )

        # Set fixed weights to match the toy weights from the issue:
        # The weights correspond to the kernel weights for forward and backward RNN layers
        # Weights format: [kernel, recurrent_kernel]
        # Here toy_weights used:
        # np.asarray([[1.0]], dtype=np.float32) for kernel forward
        # np.asarray([[-0.5]], dtype=np.float32) recurrent forward
        # np.asarray([[1.0]], dtype=np.float32) kernel backward
        # np.asarray([[-0.5]], dtype=np.float32) recurrent backward
        #
        # SimpleRNN kernel shape: (input_dim, units)
        # recurrent_kernel shape: (units, units)
        #
        # Bidirectional layers have two layers internally: forward_layer and backward_layer
        # Need to set weights on both
        
        toy_kernel = np.array([[1.0]], dtype=np.float32)
        toy_recurrent_kernel = np.array([[-0.5]], dtype=np.float32)
        toy_bias = np.zeros((1,), dtype=np.float32)  # although use_bias=False, model expects bias weights as empty or zeros

        # We will set bias to zero arrays as they are not used but layer expects full weights list.
        toy_bias_zero = np.zeros((1,), dtype=np.float32)

        # For stateless layer weights:
        # [kernel, recurrent_kernel, bias]
        stateless_fw_weights = [toy_kernel, toy_recurrent_kernel, toy_bias_zero]
        stateless_bw_weights = [toy_kernel, toy_recurrent_kernel, toy_bias_zero]
        
        # For stateful layer weights:
        stateful_fw_weights = [toy_kernel, toy_recurrent_kernel, toy_bias_zero]
        stateful_bw_weights = [toy_kernel, toy_recurrent_kernel, toy_bias_zero]

        # Set weights on forward and backward layers
        self.stateless_rnn.forward_layer.set_weights(stateless_fw_weights)
        self.stateless_rnn.backward_layer.set_weights(stateless_bw_weights)

        self.stateful_rnn.forward_layer.set_weights(stateful_fw_weights)
        self.stateful_rnn.backward_layer.set_weights(stateful_bw_weights)

    def call(self, inputs):
        """
        Forward method returns a tuple:
        (output_stateless, output_stateful_diff)
        where:
        - output_stateless: output of stateless bidirectional RNN
        - output_stateful_diff: output difference between stateful and stateless model
          (stateful model output - stateless model output)
          
        This encapsulates the observed behavior showing discrepancy between them,
        especially after reset_states() on the stateful model.
        """
        out_stateless = self.stateless_rnn(inputs)  # shape (batch, timesteps, 2*units)
        out_stateful = self.stateful_rnn(inputs)

        diff = out_stateful - out_stateless

        return out_stateless, diff

    def reset_stateful_states(self):
        """
        Reset states only of the stateful bidirectional layer.
        (As in the issue, resetting the model states is intended to reset
        stateful RNN internal states.)
        """
        self.stateful_rnn.reset_states()

def my_model_function():
    return MyModel()

def GetInput():
    # Input batch of shape (1, 3, 1) - batch_size=1, sequence_length=3, feature_dim=1
    # Following example input from the issue: first element 1, rest zeros
    x = np.zeros((1, 3, 1), dtype=np.float32)
    x[0, 0, 0] = 1.0
    return tf.constant(x)

