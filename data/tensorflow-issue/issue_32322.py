# tf.random.uniform((batch_size, sequence_length), dtype=tf.int32) ← input shape inferred from example usage with mask (e.g., embedding masking on sequences)

import tensorflow as tf

class RandomMaskingNoAlter(tf.keras.layers.Layer):
    def call(self, inputs):
        # Adding zero to inputs ensures a new tensor is returned, triggering compute_mask call
        return inputs + 0

    def compute_mask(self, inputs, mask=None):
        print('no alter executed')
        if mask is None:
            return None
        # random boolean mask same shape as input mask
        random_mask = tf.cast(tf.random.uniform(tf.shape(mask), 0, 1, dtype=tf.int32), tf.bool)
        # combine random mask with input mask
        return tf.math.logical_and(random_mask, mask)

class RandomMaskingAlter(tf.keras.layers.Layer):
    def call(self, inputs):
        # Alters inputs slightly by adding zero - triggers compute_mask
        return inputs + 0

    def compute_mask(self, inputs, mask=None):
        print('alter executed')
        if mask is None:
            return None
        random_mask = tf.cast(tf.random.uniform(tf.shape(mask), 0, 1, dtype=tf.int32), tf.bool)
        return tf.math.logical_and(random_mask, mask)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer with mask_zero=True to trigger masking
        self.embedding = tf.keras.layers.Embedding(input_dim=5, output_dim=5, mask_zero=True)
        # Instantiate the two custom masking layers
        self.random_mask_no_alter = RandomMaskingNoAlter()
        self.random_mask_alter = RandomMaskingAlter()

    def call(self, inputs):
        """
        inputs: integer tensor shape (batch_size, seq_len), e.g. token IDs
        Process inputs through embedding, then both masking layers,
        returning a boolean tensor indicating whether the outputs' masks match.
        This fuses the logic of both layers and compares their compute_mask results.
        """
        # Apply embedding to get embeddings with mask
        embedded = self.embedding(inputs)
        
        # Obtain mask propagated by embedding layer
        input_mask = embedded._keras_mask  # Using _keras_mask (private) for demonstration
        
        # Run both layers on embedded input
        # Note: these layers do not alter data meaningfully; they differ in compute_mask triggering
        out_no_alter = self.random_mask_no_alter(embedded)
        out_alter = self.random_mask_alter(embedded)
        
        # Retrieve their masks using compute_mask explicitly
        mask_no_alter = self.random_mask_no_alter.compute_mask(out_no_alter, input_mask)
        mask_alter = self.random_mask_alter.compute_mask(out_alter, input_mask)
        
        # Safely handle None masks by converting to all-True
        mask_no_alter_safe = tf.ones_like(input_mask, dtype=tf.bool) if mask_no_alter is None else mask_no_alter
        mask_alter_safe = tf.ones_like(input_mask, dtype=tf.bool) if mask_alter is None else mask_alter
        
        # Compare the two masks to see if they are equal in all positions
        masks_equal = tf.reduce_all(tf.equal(mask_no_alter_safe, mask_alter_safe))
        
        # Return the boolean value indicating whether compute_mask behavior aligns
        # This highlights the difference described in the issue:
        # the "no alter" layer's compute_mask might not be called properly without the hack
        return masks_equal

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Provide a random integer tensor simulating a batch of sequences with padding (0)
    # shape here is (3, 6) to mimic example from the issue
    # Values are token ids from 0-4 (5 vocab size, 0 is padding for mask_zero)
    x = tf.constant([
        [1, 4, 2, 2, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [3, 2, 2, 3, 4, 1]
    ], dtype=tf.int32)
    return x

