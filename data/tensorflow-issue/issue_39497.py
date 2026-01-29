# tf.random.uniform((B, T), dtype=tf.int32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size=1000, embedding_dim=300, num_tags=3, cell_units=300):
        super().__init__()
        # Embedding layer with mask_zero=True to support variable length sequences
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        
        # Create forward and backward RNN layers locally (not assigned as attributes)
        forward_rnn_layer = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(cell_units),
            return_sequences=True,
            return_state=False
        )
        backward_rnn_layer = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(cell_units),
            return_sequences=True,
            return_state=False,
            go_backwards=True
        )
        
        # Bidirectional wrapper using the above forward and backward layers
        self.bi_rnn_layer = tf.keras.layers.Bidirectional(
            forward_rnn_layer,
            backward_layer=backward_rnn_layer
        )
        
        # Final dense layer to output tag logits for each timestep
        self.fc = tf.keras.layers.Dense(num_tags)

    def call(self, inputs, sequence_length):
        """
        Args:
          inputs: integer tensor of shape [batch_size, sequence_length], token ids
          sequence_length: integer tensor of shape [batch_size], lengths of sequences
      
        Returns:
          Output tensor of shape [batch_size, sequence_length, num_tags],
          logits per timestep.
        """
        # Apply embedding
        x = self.embedding(inputs)  # shape: [B, T, embedding_dim]
        
        # Create mask based on sequence_length for padded timesteps
        mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1])
        
        # Apply bidirectional RNN with mask
        x = self.bi_rnn_layer(x, mask=mask)
        
        # Final dense layer to compute logits
        logits = self.fc(x)
        
        return logits

def my_model_function():
    # Instantiate model with default hyperparameters
    return MyModel()

def GetInput():
    """
    Returns:
      A tuple (inputs, sequence_length) suitable for feeding into MyModel.
      - inputs: int32 tensor of shape [batch_size, max_seq_len], random token IDs
      - sequence_length: int32 tensor of shape [batch_size], lengths <= max_seq_len
      
    Assumptions:
      - Batch size 4, max_seq_len 7 (arbitrary reasonable choice)
      - Vocabulary size 1000 (matching default in model)
      - Sequence lengths randomly chosen between 1 and max_seq_len
    """
    batch_size = 4
    max_seq_len = 7
    vocab_size = 1000
    
    # Random integer tokens in [1, vocab_size-1], reserve 0 for padding
    inputs = tf.random.uniform(
        shape=(batch_size, max_seq_len),
        minval=1,
        maxval=vocab_size,
        dtype=tf.int32
    )
    
    # Random sequence lengths between 1 and max_seq_len inclusive
    sequence_length = tf.random.uniform(
        shape=(batch_size,),
        minval=1,
        maxval=max_seq_len + 1,
        dtype=tf.int32
    )
    
    # Zero out tokens beyond sequence_length to simulate padding correctly
    # For each sample in batch, mask positions beyond length with 0
    padding_mask = tf.sequence_mask(sequence_length, maxlen=max_seq_len)
    inputs = tf.where(padding_mask, inputs, tf.zeros_like(inputs))
    
    return inputs, sequence_length

