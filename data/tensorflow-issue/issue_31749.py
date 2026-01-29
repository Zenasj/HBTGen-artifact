# tf.random.normal(shape=(dynamic_size,)) where dynamic_size is determined by input scalar tensor shape=(1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable layers, output shape depends on input scalar values
        
    def call(self, inputs):
        # inputs shape: (batch_size, 1), dtype int32 or int64 assumed
        # For each batch element, produce output vector of length = inputs[i, 0]
        # Since TensorFlow graph requires static shapes for outputs in tf.function,
        # a workaround is required. Example here produces ragged outputs for demonstration.
        # However, for compatibility with loss, we generate a padded output with zeros.
        
        # Extract 1D vector of lengths:
        lengths = tf.reshape(inputs, [-1])  # shape (batch_size,)
        
        max_len = tf.reduce_max(lengths)  # scalar scalar for padding

        # Generate per-batch outputs of shape (batch_size, max_len)
        # For positions >= input length, output zero logits (so loss ignores?)
        def gen_logits(length):
            # Random normal outputs of shape (length,)
            logits = tf.random.normal(shape=(length,), dtype=tf.float32)
            return logits

        # Use tf.ragged to generate ragged tensor of outputs
        ragged_logits = tf.RaggedTensor.from_row_lengths(
            values=tf.concat([gen_logits(length) for length in lengths], axis=0),
            row_lengths=lengths)
        # Can't directly return ragged from tf.keras.Model, so pad to max_len
        padded_logits = ragged_logits.to_tensor(shape=[None, max_len], default_value=0.0)
        
        # For downstream loss compatibility, outputs length must match labels
        # so user must prepare labels padded to max_len as well
        
        return padded_logits


def my_model_function():
    # Return an instance of the model
    return MyModel()


def GetInput():
    # Generate sample input tensor matching input expectations:
    # According to example, input shape = (batch_size, 1), dtype int32, values between 0 and 10
    batch_size = 5
    # Random integers shape=(batch_size,1), values 0 to 10 inclusive
    inputs = tf.random.uniform((batch_size,1), minval=0, maxval=11, dtype=tf.int32)
    return inputs

