# tf.random.uniform((B, T, V), dtype=tf.float32) ← Input shape: (batch_size, time_steps, vocab_size=number_of_words)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_words=10000, time_steps=67, embedding_dim=256, gru_units=512):
        """
        A sequence-to-sequence model for machine translation based on the issue discussion.
        Encoder and decoder use Embedding + GRU layers.
        
        Args:
            num_words: Vocabulary size for output dense layer.
            time_steps: Fixed length of target sequences.
            embedding_dim: Dimension of embeddings.
            gru_units: Number of units in the GRU layers.
        """
        super().__init__()
        # Encoder embedding and GRU
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=num_words,
                                                           output_dim=embedding_dim,
                                                           mask_zero=True,
                                                           name="encoder_embedding")
        self.encoder_gru = tf.keras.layers.GRU(gru_units,
                                               return_state=True,
                                               return_sequences=False,
                                               name="encoder_gru")
        
        # Decoder embedding and GRU
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=num_words,
                                                           output_dim=embedding_dim,
                                                           mask_zero=True,
                                                           name="decoder_embedding")
        self.decoder_gru = tf.keras.layers.GRU(gru_units,
                                               return_sequences=True,
                                               return_state=True,
                                               name="decoder_gru")
        # Final Dense layer with linear activation -- will use TF sparse softmax cross-entropy loss with logits
        self.decoder_dense = tf.keras.layers.Dense(num_words,
                                                   activation='linear',  # No softmax here, loss fn expects logits
                                                   name="decoder_output")
        
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: tuple/list of (encoder_input, decoder_input)
                - encoder_input: tf.int32 tensor of shape (batch_size, encoder_seq_len)
                - decoder_input: tf.int32 tensor of shape (batch_size, decoder_seq_len)
        
        Returns:
            logits of shape (batch_size, decoder_seq_len, num_words)
        """
        encoder_input, decoder_input = inputs
        
        # Encoder
        enc_emb = self.encoder_embedding(encoder_input)  # (B, E_seq, embed_dim)
        enc_output, enc_state = self.encoder_gru(enc_emb)  # enc_state shape: (B, gru_units)
        
        # Decoder
        dec_emb = self.decoder_embedding(decoder_input)  # (B, T, embed_dim)
        # Using encoder final state as initial state for decoder gru
        dec_output, _ = self.decoder_gru(dec_emb, initial_state=enc_state)  # (B, T, gru_units)
        
        logits = self.decoder_dense(dec_output)  # (B, T, num_words)
        
        return logits

def sparse_loss(y_true, y_pred):
    """
    Custom sparse categorical crossentropy loss using TensorFlow's
    sparse_softmax_cross_entropy_with_logits.
    
    Keras expects loss functions to return a tensor with per-sample loss.
    This function does not reduce along time or batch dims to keep masking valid.
    
    Args:
        y_true: true labels tensor of shape (batch_size, time_steps, 1) and dtype int32 or int64
        y_pred: logits tensor of shape (batch_size, time_steps, num_words)
    
    Returns:
        Tensor of shape (batch_size, time_steps) representing per-timestep loss.
    """
    # Remove the last dimension from y_true (squeeze size-1 dimension)
    y_true = tf.squeeze(y_true, axis=-1)
    # Compute loss per element
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)  # shape (B, T)
    return loss  # not reduced; reduction handled by Keras internally

def my_model_function():
    """
    Create and return an instance of MyModel.
    Uses default hyperparameters from the issue context.
    """
    # Assume typical vocab size (e.g. 10,000) and fixed decoder sequence length 67 as per issue.
    model = MyModel(num_words=10000, time_steps=67)
    # Compile model with optimizer and the custom sparse loss function
    model.compile(optimizer='adam',
                  loss=sparse_loss,
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

def GetInput():
    """
    Returns random inputs suitable for MyModel.
    Inputs are integer token sequences for encoder and decoder of fixed lengths.
    
    Encoder input shape: (batch_size, encoder_seq_len)
    Decoder input shape: (batch_size, decoder_seq_len)
    
    We'll use:
        - batch_size = 4 (arbitrary)
        - encoder sequence length = 50 (arbitrary)
        - decoder sequence length = 67 (from issue)
        - vocab indices from [0, num_words-1]
    
    Returns:
        Tuple of (encoder_input, decoder_input), both tf.Tensor of dtype tf.int32
    """
    batch_size = 4
    encoder_seq_len = 50  # arbitrary length for encoder input
    decoder_seq_len = 67  # fixed length from issue
    num_words = 10000
    
    # Random integer token inputs in range [1, num_words - 1] (reserve 0 for padding)
    encoder_input = tf.random.uniform(shape=(batch_size, encoder_seq_len),
                                      minval=1,
                                      maxval=num_words,
                                      dtype=tf.int32)
    decoder_input = tf.random.uniform(shape=(batch_size, decoder_seq_len),
                                      minval=1,
                                      maxval=num_words,
                                      dtype=tf.int32)
    return (encoder_input, decoder_input)

