# tf.random.uniform((8, 64, 256), dtype=tf.float32) ← input shape inferred from example: batch=8, seq_len=64, features=256

import tensorflow as tf

class SafeRNN(tf.keras.layers.Wrapper):
    """Wrapper for keras RNN layers avoiding a mask-caused CUDA cuDNN error when all-mask sequences occur on GPU."""
    
    def call(self, inputs, mask=None, **kwargs):
        """Run inputs through the wrapped RNN layer safely with adjusted mask.
        
        If any sequence in the batch is completely masked out (all False),
        replace its mask with all True to avoid cuDNN error on GPU.
        """
        if mask is not None:
            # For each sequence in batch, check if it has any True in mask along time dimension
            valid = tf.reduce_any(mask, axis=1, keepdims=True)  # shape: (batch, 1)
            # If not valid (all False), replace mask for that sequence with all True
            mask = tf.where(valid, mask, tf.ones_like(mask))
        return self.layer(inputs, mask=mask, **kwargs)

    def compute_mask(self, inputs, mask=None):
        """Compute reduced mask after processing.

        Simplify mask by reducing time dimension using any().
        This represents whether each sequence has any valid timestep left.
        """
        if mask is None:
            return None
        return tf.reduce_any(mask, axis=1)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate the wrapped GRU layer with 128 units inside SafeRNN to avoid cuDNN error
        self.safe_gru = SafeRNN(tf.keras.layers.GRU(128))
        
    def call(self, inputs, mask=None):
        """Forward pass of MyModel."""
        return self.safe_gru(inputs, mask=mask)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input and a mask that replicates the minimal example scenario:
    # batch=8, seq_len=64, features=256
    # mask has first 4 sequences fully unmasked (True), last 4 fully masked (False)
    rng = tf.random.get_global_generator()
    inp = rng.normal((8, 64, 256), dtype=tf.float32)
    msk = tf.concat([tf.ones((4, 64), dtype=tf.bool), tf.zeros((4, 64), dtype=tf.bool)], axis=0)
    return inp, msk

