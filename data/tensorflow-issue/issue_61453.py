# tf.random.uniform((1, 15), dtype=tf.int32) ← Inferred input shape from padded sequences (max_seq_length=15)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using same tokenizer vocab sizes and latent dim as original code inference
        # For demonstration, vocab sizes are set to realistic placeholder values:
        # (In a real use case, these would be passed or loaded from tokenizer)
        vocab_size_encoder = 30  # example vocab size for input_tokens (word_index + 1)
        vocab_size_decoder = 40  # example vocab size for output_tokens (word_index + 1)
        latent_dim = 64
        self.max_seq_length = 15  # inferred from the issue's padding length

        # Encoder layers
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size_encoder,
                                                           output_dim=latent_dim,
                                                           mask_zero=True)
        self.encoder_lstm = tf.keras.layers.LSTM(latent_dim, return_state=True)

        # Decoder layers
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size_decoder,
                                                           output_dim=latent_dim,
                                                           mask_zero=True)
        self.decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
        self.decoder_dense = tf.keras.layers.Dense(vocab_size_decoder, activation='softmax')

    def call(self, inputs, training=False):
        # inputs is a list or tuple: [encoder_input_seq, decoder_input_seq]
        encoder_inputs, decoder_inputs = inputs
        
        # Encoder forward pass
        encoder_emb = self.encoder_embedding(encoder_inputs)  # (batch, seq_len, latent_dim)
        _, state_h, state_c = self.encoder_lstm(encoder_emb, training=training)
        encoder_states = [state_h, state_c]

        # Decoder forward pass
        decoder_emb = self.decoder_embedding(decoder_inputs)  # (batch, seq_len, latent_dim)
        decoder_outputs, _, _ = self.decoder_lstm(decoder_emb,
                                                  initial_state=encoder_states,
                                                  training=training)  # (batch, seq_len, latent_dim)
        outputs = self.decoder_dense(decoder_outputs)  # (batch, seq_len, vocab_size_decoder)

        return outputs

def my_model_function():
    # Returns an instance of MyModel, initialize with default sizes matching inference above
    return MyModel()

def GetInput():
    # Returns a tuple of two input tensors matching model expected input shapes:
    # encoder input shape: (batch=1, sequence_length=15) with integer token IDs >=0 and < vocab_size_encoder
    # decoder input shape: (batch=1, sequence_length=15) similarly integer tokens
    batch_size = 1
    seq_len = 15
    vocab_size_encoder = 30
    vocab_size_decoder = 40
    
    # Random tokens including zero (mask token) and 1 .. vocab_size - 1
    encoder_input = tf.random.uniform(shape=(batch_size, seq_len),
                                      minval=0,
                                      maxval=vocab_size_encoder,
                                      dtype=tf.int32)
    decoder_input = tf.random.uniform(shape=(batch_size, seq_len),
                                      minval=0,
                                      maxval=vocab_size_decoder,
                                      dtype=tf.int32)
    return (encoder_input, decoder_input)

