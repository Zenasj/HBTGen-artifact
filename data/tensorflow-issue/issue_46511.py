# tf.random.uniform((B, T, 5), dtype=tf.float32) ← B=batch size, T=sequence length, 5=number of labels

import tensorflow as tf
import tensorflow_addons as tfa

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Transition matrix for CRF of shape (num_tags, num_tags)
        self.num_tags = 5
        # Initialize transition_params as a trainable weight
        self.transition_params = self.add_weight(
            shape=(self.num_tags, self.num_tags),
            regularizer=tf.keras.regularizers.l2(0.1),
            name="transitions",
            trainable=True
        )

    def call(self, potentials, sequence_lengths, tag_indices):
        """
        potentials: Float tensor of shape (batch_size, max_seq_len, num_tags)
        sequence_lengths: Int tensor of shape (batch_size,)
        tag_indices: Int tensor of shape (batch_size, max_seq_len), for log likelihood calc
        """
        # Decode the best sequence using CRF decode
        # Returns (decoded_sequences, potentials)
        decoded_sequence, _ = tfa.text.crf.crf_decode(
            potentials, self.transition_params, sequence_lengths
        )
        
        # Compute log likelihood of the gold sequence with CRF log likelihood function
        log_likelihood, _ = tfa.text.crf.crf_log_likelihood(
            potentials, tag_indices, sequence_lengths, self.transition_params
        )
        
        return decoded_sequence, log_likelihood

def my_model_function():
    # Return an instance of MyModel with trainable transitions
    return MyModel()

def GetInput():
    """
    Returns a tuple of inputs that can be directly fed to MyModel.call:
    - potentials: random tensor (batch_size, seq_len, num_tags)
    - sequence_lengths: vector of sequence lengths (batch_size,)
    - tag_indices: random integer tags (batch_size, seq_len)
    
    We assume batch_size=3, seq_len=4, num_tags=5 for example.
    This matches the example code in the issue.
    """
    batch_size = 3
    seq_len = 4
    num_tags = 5
    
    # potentials: random floats
    potentials = tf.random.uniform((batch_size, seq_len, num_tags), dtype=tf.float32)
    # sequence lengths: for simplicity, all full length in this example
    sequence_lengths = tf.constant([seq_len] * batch_size, dtype=tf.int32)
    # random tag indices between 0..num_tags-1, dtype int32
    tag_indices = tf.random.uniform((batch_size, seq_len), minval=0, maxval=num_tags, dtype=tf.int32)
    
    return potentials, sequence_lengths, tag_indices

