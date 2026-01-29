# tf.random.uniform((2, 3, 5), dtype=tf.float32) ← batch=2, sequence_length=3, feature_size=5

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # GRU layer from the reported issue with return_sequences=True and return_state=True
        self.gru = tf.keras.layers.GRU(10, return_sequences=True, return_state=True)

    def call(self, inputs):
        # Call the GRU layer, which returns a tuple of (output_sequence, last_state)
        # The original issue noted a discrepancy in GPU output shapes:
        #   CPU output_state shape: [batch_size, units]  (expected)
        #   GPU output_state shape: [units]              (unexpected)
        #
        # To unify behavior, we explicitly check and reshape output_state if needed to
        # ensure output_state always has shape [batch_size, units].
        outputs = self.gru(inputs)
        output_sequence, output_state = outputs[0], outputs[1]

        # If output_state rank is 1 (likely GPU case), add batch dimension
        if tf.rank(output_state) == 1:
            # Assume batch size is inputs.shape[0], expand dims
            batch_size = tf.shape(inputs)[0]
            output_state = tf.expand_dims(output_state, axis=0)

            # If batch_size > 1 and output_state batch size == 1, we tile it
            # to match batch size to prevent shape mismatches downstream.
            # This is an inferred workaround for the GPU behavior discrepancy.
            output_state = tf.tile(output_state, multiples=[batch_size, 1])

        # Return unified output, matching the CPU's expected shape for output_state
        return output_sequence, output_state

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching expected input shape: (batch=2, seq_len=3, feat=5)
    return tf.random.uniform((2, 3, 5), dtype=tf.float32)

