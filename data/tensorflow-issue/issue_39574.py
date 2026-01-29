# tf.random.uniform((B, 200, 311), dtype=tf.float32) ← inferred input shape and dtype from original data X (35000, 200, 311)

import tensorflow as tf
from tensorflow.keras.layers import Masking, Bidirectional, GRU, TimeDistributed, Dense

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Mask input where features are zero (mask_value=0.0)
        self.masking = Masking(mask_value=0.0)
        
        # Bidirectional GRU layer with 256 units, returns sequences for each timestep
        # The original used unroll=True, recurrent_dropout=0.233, recurrent_activation='sigmoid'
        # Note: recurrent_dropout is generally only effective during training and may not be ideal for XLA,
        # but we keep as specified.
        self.bi_gru = Bidirectional(
            GRU(
                256,
                return_sequences=True,
                unroll=True,
                recurrent_dropout=0.233,
                recurrent_activation="sigmoid"
            )
        )
        
        # TimeDistributed Dense layer to produce a softmax over 19 classes per timestep
        self.td_dense = TimeDistributed(Dense(19, activation="softmax"))
    
    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        Args:
          inputs: tensor of shape (batch_size, 200, 311)
          training: bool, whether in training mode (affects dropout)
        Returns:
          tensor of shape (batch_size, 200, 19) with softmax probabilities per timestep
        """
        x = self.masking(inputs)
        x = self.bi_gru(x, training=training)
        x = self.td_dense(x)
        return x

def my_model_function():
    """
    Returns:
      An instance of the MyModel class with all layers initialized.
    """
    return MyModel()

def GetInput():
    """
    Returns:
      A random tensor input matching the expected input shape for the model:
      A float32 tensor with shape (batch_size, 200, 311).
      Batch size chosen reasonably small (e.g., 16) to reflect training batch size.
    """
    batch_size = 16  # batch size used in original training code
    # Generate random input in [0, 1) as float32 tensor
    return tf.random.uniform((batch_size, 200, 311), dtype=tf.float32)

