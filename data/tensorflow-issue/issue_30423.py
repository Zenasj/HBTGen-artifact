# tf.random.uniform((2, 5), dtype=tf.float32) ← Input shape inferred from the reproducing code: batch=2, timesteps=5

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding with mask_zero=True to produce mask for zero input tokens
        self.embedding = tf.keras.layers.Embedding(input_dim=16, output_dim=5, mask_zero=True)
        
        # Two GRU layers, instantiated with both zero_output_for_mask True and False,
        # to compare their output behavior on masked sequences.
        self.gru_zero_mask = tf.keras.layers.GRU(5, return_sequences=True, zero_output_for_mask=True)
        self.gru_prev_output_mask = tf.keras.layers.GRU(5, return_sequences=True, zero_output_for_mask=False)
    
    def call(self, inputs, training=None):
        """
        Forward pass:
        Compare the output of GRU with zero_output_for_mask True vs False when mask zeros appear.
        
        Returns:
          A dict with:
            - outputs_zero_mask: GRU output with zero_output_for_mask=True
            - outputs_prev_mask: GRU output with zero_output_for_mask=False
            - equal_masked: tf.Tensor of shape (batch, timesteps, units) boolean indicating
                            where outputs are equal between the two modes
            - difference: numeric difference between the two outputs
        """
        x = self.embedding(inputs)  # Embedding layer outputs (batch, timesteps, emb_dim)
        
        # Compute both versions of GRU outputs
        out_zero_mask = self.gru_zero_mask(x, training=training)    # zeros on masked steps
        out_prev_mask = self.gru_prev_output_mask(x, training=training)  # previous outputs on masked steps
        
        # Compare outputs to show difference at masked positions
        difference = out_zero_mask - out_prev_mask
        equal_masked = tf.math.equal(out_zero_mask, out_prev_mask)
        
        return {
            'outputs_zero_mask': out_zero_mask,
            'outputs_prev_mask': out_prev_mask,
            'equal_masked': equal_masked,
            'difference': difference
        }

def my_model_function():
    # Return an instance of MyModel with initialized weights (default init)
    return MyModel()

def GetInput():
    # Provide a batch of two input sequences (dtype tf.int32) with masking zeros at some timesteps
    import numpy as np

    # Batch size 2, sequence length 5, values from 1 to 15, zeros for masked.
    np_x = np.ones((2, 5), dtype=np.int32)
    # Mask last two timesteps of second sample by zero token index (0 is special zero-masked token in Embedding)
    np_x[1, 3:] = 0

    return tf.convert_to_tensor(np_x)

