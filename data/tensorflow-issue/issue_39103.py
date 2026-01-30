import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout

DROPOUT = 0.0
LSTM_DIM = 512

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
            self.segment_embedding = Embedding(3, segment_embedding_dim, weights=[np.zeros((3, segment_embedding_dim))], trainable=False, mask_zero=True, name="segment_embedding")
        
        self.lstm1 = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="enc_lstm1")
        
    @tf.function
    def call(self, inputs):
        input_utterence, segment_tokens, initial_state = inputs
        input_embed = self.embedding(input_utterence)
        segment_embed = self.segment_embedding(segment_tokens)
        
        combined_embed = tf.add(input_embed, segment_embed)
        
        encoder_states, h1, c1 = self.lstm1(combined_embed, initial_state=[initial_state, initial_state])
        
        return encoder_states, h1, c1
    
    def create_initial_state(self):
        return tf.zeros((self.batch_size, self.n_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding, n_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.n_units = n_units
        
        self.embedding = embedding
        
        self.lstm1 = LSTM(n_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform", name="dec_lstm1")
        
        self.dropout = Dropout(DROPOUT)
        
        # attention
        # Ct(s) = V tanh(W1 hs + W2 ht)
        # where hs is encoder state at timestep s and ht is the previous
        # decoder timestep (which is at timestep t - 1)
        self.W1 = Dense(n_units)
        self.W2 = Dense(n_units)
        self.V  = Dense(1)
        
        # from_logits=True in loss function, it will apply the softmax there for us
        self.out_dense1 = Dense(vocab_size)
    
    @tf.function
    def call(self, inputs):
        input_word, encoder_outputs, is_training, hidden = inputs
        h1, c1 = hidden
        
        # ------ Attention ------ #
        # => (batch_size, 1, n_units)
        decoder_state = tf.expand_dims(h1, 1)
        
        # score shape => (batch_size, src_timesteps, 1)
        score = self.V(
            tf.nn.tanh(self.W1(encoder_outputs) + self.W2(decoder_state)) )
        
        attn_weights = tf.nn.softmax(score, axis=1)
        
        # context vector is a weighted sum of attention weights with encoder outputs
        context_vec = attn_weights * encoder_outputs
        # => (batch_size, n_units)
        context_vec = tf.reduce_sum(context_vec, axis=1)
        # ------ ------ #
        
        input_embed = self.embedding(input_word)
        
        # feed context vector as input into LSTM at current timestep
        input_embed = tf.concat([tf.expand_dims(context_vec, 1), input_embed], axis=-1)
        
        decoder_output, h1, c1 = self.lstm1(input_embed, initial_state=[h1, c1])
        
        # (batch_size, 1, n_units) => (batch_size, n_units)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        
        decoder_output = self.dropout(decoder_output, training=is_training)
        decoder_output = self.out_dense1(decoder_output)
        
        return decoder_output, attn_weights, h1, c1

vocab_size = 100

embedding_matrix = Embedding(vocab_size, 300, trainable=True, mask_zero=True, name="tied_embedding")

encoder = Encoder(vocab_size, embedding_matrix, LSTM_DIM, 64, True, 300)
decoder = Decoder(vocab_size, embedding_matrix, LSTM_DIM, 64)

enc_state = enc_state = encoder.create_initial_state()
out, h1, c1 = encoder([tf.random.uniform((64, 10), 0, 99),
        tf.random.uniform((64, 10), 0, 2),
        enc_state])

decoder([tf.random.uniform((64, 1), 0, 99), out, True, [h1, c1]])

encoder_fn = "seq2seq_encoder"
decoder_fn = "seq2seq_decoder"
decoder_states_spec = [
            tf.TensorSpec(shape=[None, LSTM_DIM], dtype=tf.float32, name='h1'), 
            tf.TensorSpec(shape=[None, LSTM_DIM], dtype=tf.float32, name='c1')]
        
tf.saved_model.save(encoder, encoder_fn , signatures=encoder.call.get_concrete_function(
        [
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input_utterence'),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='segment_tokens'),
            tf.TensorSpec(shape=[None, 512], dtype=tf.float32, name="initial_state")
        ]
))
    
tf.saved_model.save(decoder, decoder_fn, signatures=decoder.call.get_concrete_function(
        [
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input_word'), 
            tf.TensorSpec(shape=[None, None, LSTM_DIM], dtype=tf.float32, name="encoder_output"),
            tf.TensorSpec(shape=[], dtype=tf.bool, name="is_training"),
            decoder_states_spec
        ]
))

encoder = tf.saved_model.load(encoder_fn)
decoder = tf.saved_model.load(decoder_fn)

enc_state = enc_state = tf.zeros((64, LSTM_DIM))
out, h1, c1 = encoder([tf.random.uniform((64, 10), 0, 99),
        tf.random.uniform((64, 10), 0, 2),
        enc_state])

decoder([tf.random.uniform((64, 1), 0, 99), out, True, [h1, c1]])