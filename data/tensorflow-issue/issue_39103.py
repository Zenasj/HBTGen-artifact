# tf.random.uniform((64, 10), dtype=tf.int32) and tf.random.uniform((64, 1), dtype=tf.int32)
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
import numpy as np

DROPOUT = 0.0
LSTM_DIM = 512
VOCAB_SIZE = 100
BATCH_SIZE = 64
SEGMENT_EMBED_DIM = 300

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding, n_units, batch_size, use_segment_embedding, segment_embedding_dim):
        super(Encoder, self).__init__()
        self.n_units = n_units
        self.batch_size = batch_size
        
        self.embedding = embedding
        
        # segment embedding are used so that this model can better distinguish between persona and message segments
        # pad segment vectors with 0's exactly like word vectors
        if use_segment_embedding:
            # segment_embedding_dim must be the same as output_dim of word embedding
            self.segment_embedding = Embedding(3, segment_embedding_dim, trainable=True, mask_zero=True, name="segment_embedding")
        else:
            # use a zero segment embedding which will have no effect on the model
            zero_weights = [np.zeros((3, segment_embedding_dim))]
            self.segment_embedding = Embedding(3, segment_embedding_dim, weights=zero_weights, trainable=False, mask_zero=True, name="segment_embedding")
        
        self.lstm1 = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_lstm1")
        
    @tf.function
    def call(self, inputs):
        # inputs: (input_utterance: int32 tensor [batch, seq_len],
        #          segment_tokens: int32 tensor [batch, seq_len],
        #          initial_state: float32 tensor [batch, n_units])
        input_utterance, segment_tokens, initial_state = inputs
        input_embed = self.embedding(input_utterance)  # [batch, seq_len, emb_dim]
        segment_embed = self.segment_embedding(segment_tokens)  # [batch, seq_len, emb_dim]
        
        combined_embed = tf.add(input_embed, segment_embed)  # [batch, seq_len, emb_dim]
        
        encoder_outputs, h1, c1 = self.lstm1(combined_embed, initial_state=[initial_state, initial_state])
        
        return encoder_outputs, h1, c1
    
    def create_initial_state(self):
        # zeros tensor of shape [batch_size, n_units]
        return tf.zeros((self.batch_size, self.n_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.n_units = n_units
        
        self.embedding = embedding
        
        self.lstm1 = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_lstm1")
        
        self.dropout = Dropout(DROPOUT)
        
        # attention mechanism components
        # Ct(s) = V tanh(W1 hs + W2 ht)
        self.W1 = Dense(n_units)
        self.W2 = Dense(n_units)
        self.V  = Dense(1)
        
        # Dense layer to produce logits over vocabulary
        self.out_dense1 = Dense(vocab_size)
    
    @tf.function
    def call(self, inputs):
        # inputs: (input_word: int32 [batch, 1],
        #          encoder_outputs: float32 [batch, src_timesteps, n_units],
        #          is_training: bool scalar,
        #          hidden: list of two float32 tensors [h1, c1] each [batch, n_units])
        input_word, encoder_outputs, is_training, hidden = inputs
        h1, c1 = hidden
        
        # --- Attention ---
        decoder_state = tf.expand_dims(h1, 1)  # [batch, 1, n_units]
        
        # Score calculation: [batch, src_timesteps, 1]
        score = self.V(tf.nn.tanh(self.W1(encoder_outputs) + self.W2(decoder_state)))
        
        attn_weights = tf.nn.softmax(score, axis=1)  # [batch, src_timesteps, 1]
        
        # Context vector: weighted sum of encoder outputs [batch, n_units]
        context_vec = attn_weights * encoder_outputs
        context_vec = tf.reduce_sum(context_vec, axis=1)
        
        # Embedding for input word: [batch, 1, emb_dim]
        input_embed = self.embedding(input_word)
        
        # Concatenate context vector and input embedding on last axis: [batch, 1, emb_dim + n_units]
        input_embed = tf.concat([tf.expand_dims(context_vec, 1), input_embed], axis=-1)
        
        decoder_output, h1, c1 = self.lstm1(input_embed, initial_state=[h1, c1])
        
        # Reshape output to [batch, n_units]
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        
        decoder_output = self.dropout(decoder_output, training=is_training)
        decoder_output = self.out_dense1(decoder_output)  # logits
        
        return decoder_output, attn_weights, h1, c1


# Fuse Encoder and Decoder as submodules in MyModel with a comparison example
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Shared embedding layer (trainable)
        self.embedding = Embedding(VOCAB_SIZE, SEGMENT_EMBED_DIM, trainable=True, mask_zero=True, name="shared_embedding")
        
        # Encoder and Decoder submodels
        self.encoder = Encoder(VOCAB_SIZE, self.embedding, LSTM_DIM, BATCH_SIZE, True, SEGMENT_EMBED_DIM)
        self.decoder = Decoder(VOCAB_SIZE, self.embedding, LSTM_DIM, BATCH_SIZE)
    
    @tf.function
    def call(self, inputs):
        """
        inputs: tuple of 
          (encoder_input_utterance: int32 [batch, seq_len],
           encoder_segment_tokens: int32 [batch, seq_len],
           decoder_input_word: int32 [batch, 1],
           is_training: bool scalar)
        Returns:
          (decoder_output_logits: float32 [batch, vocab_size],
           attention_weights: float32 [batch, src_timesteps, 1],
           encoder_hidden_state: float32 [batch, n_units],
           decoder_hidden_state: float32 [batch, n_units])
        """
        encoder_input_utterance, encoder_segment_tokens, decoder_input_word, is_training = inputs
        
        # Initialize encoder initial state
        enc_init_state = self.encoder.create_initial_state()
        
        # Run encoder
        encoder_outputs, enc_h, enc_c = self.encoder([encoder_input_utterance, encoder_segment_tokens, enc_init_state])
        
        # Run decoder using encoder outputs and last encoder states as initial hidden state
        decoder_outputs, attn_weights, dec_h, dec_c = self.decoder([decoder_input_word, encoder_outputs, is_training, [enc_h, enc_c]])
        
        # Example comparison: check if decoder hidden matches encoder hidden within a tolerance (just a demo)
        # Returns a bool tensor [batch] indicating per-batch exact match within tolerance.
        # This is just demonstration logic per the fusion requirement:
        compare_hidden = tf.reduce_all(tf.abs(dec_h - enc_h) < 1e-5, axis=1)
        
        # Return all outputs plus the comparison boolean
        return decoder_outputs, attn_weights, enc_h, dec_h, compare_hidden


def my_model_function():
    # Return an instance of MyModel with initialized weights
    return MyModel()

def GetInput():
    # Generate example inputs consistent with MyModel's expected input:
    # encoder_input_utterance: int32 [batch, seq_len] (e.g. 64x10)
    # encoder_segment_tokens: int32 [batch, seq_len] (e.g. 64x10 with segment ids 0-2)
    # decoder_input_word: int32 [batch, 1] (e.g. 64x1)
    # is_training: bool scalar
    encoder_input_utterance = tf.random.uniform((BATCH_SIZE, 10), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
    encoder_segment_tokens = tf.random.uniform((BATCH_SIZE, 10), minval=0, maxval=3, dtype=tf.int32)
    decoder_input_word = tf.random.uniform((BATCH_SIZE, 1), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
    is_training = tf.constant(True)
    return (encoder_input_utterance, encoder_segment_tokens, decoder_input_word, is_training)

