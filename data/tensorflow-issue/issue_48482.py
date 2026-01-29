# tf.random.uniform((batch_size=1, sequence_length=300), dtype=tf.int32) for word input
# tf.random.uniform((1, 300, 20), dtype=tf.int32) for char input
# tf.random.uniform((1, 300, 4), dtype=tf.float32) for pos input
# tf.random.uniform((1, 300, 3), dtype=tf.float32) for par input

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed, Dense, SpatialDropout1D, concatenate

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Word embedding: vocab size 1834, embedding dim 16, input_length=300, mask_zero True
        self.emb_wor = Embedding(input_dim=1834, output_dim=16, input_length=300, mask_zero=True, name="emb_wor")
        # Char embedding + encoding:
        # TimeDistributed Embedding: vocab size 132, embedding dim 32, mask_zero True
        self.emb_char_td = TimeDistributed(
            Embedding(input_dim=132, output_dim=32, input_length=20, mask_zero=True, name="emb_char"),
            name="td_emb_char"
        )
        # TimeDistributed LSTM for char encoding, units=32, return_sequences=False
        # Use recurrent_dropout=0.15 as in original
        self.char_enc_td = TimeDistributed(
            LSTM(units=32, return_sequences=False, recurrent_dropout=0.15, name="char_enc"),
            name="td_char_enc"
        )
        # No embedding layers for pos and par inputs, directly consumed

        # Spatial dropout 1D with 0.1 rate
        self.dropout = SpatialDropout1D(0.1)
        # Bidirectional LSTM on concatenated inputs:
        # units=64, return_sequences=True, dropout=0., recurrent_dropout=0.1
        self.main_lstm = Bidirectional(
            LSTM(units=64, return_sequences=True, dropout=0., recurrent_dropout=0.1, name="main_lstm")
        )
        # TimeDistributed Dense layer with 4 units and softmax activation for output
        self.out_td = TimeDistributed(Dense(4, activation="softmax", name="out"))

    def call(self, inputs):
        # inputs is a list/tuple of 4 tensors:
        # [word_in shape=(batch,300)
        #  char_in shape=(batch,300,20)
        #  input_pos shape=(batch,300,4)
        #  input_par shape=(batch,300,3)]
        word_in, char_in, input_pos, input_par = inputs

        # word embedding (batch,300,16)
        emb_wor = self.emb_wor(word_in)

        # char embedding followed by char encoding: 
        # for char_in (batch,300,20) --> emb_char_td --> (batch,300,20,32) --> char_enc_td --> (batch,300,32)
        emb_char = self.emb_char_td(char_in)
        char_enc = self.char_enc_td(emb_char)

        # concatenate along last axis, so shape: (batch,300, 16 + 32 + 4 + 3) = (batch,300,55)
        x = concatenate([emb_wor, char_enc, input_pos, input_par], axis=-1)

        # spatial dropout 1D on sequence
        x = self.dropout(x)

        # main bi-lstm layer (batch,300,128)
        x = self.main_lstm(x)

        # time distributed output dense softmax (batch,300,4)
        output = self.out_td(x)
        return output


def my_model_function():
    # Return an instance of MyModel, compiled as in original example for completeness
    model = MyModel()
    # Note: The original code used Model subclassing with compile. Here compile is optional depending on usage.
    # Add a dummy compile to match original:
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model

def GetInput():
    # Create input tensors matching the expected input shapes:
    # word_in: (batch, 300) integer indices from [0,1834)
    # char_in: (batch,300,20) integer indices from [0,132)
    # input_pos: (batch,300,4) float32 (positional or other features)
    # input_par: (batch,300,3) float32 (parent or other features)

    batch_size = 1
    word_input = tf.random.uniform(shape=(batch_size, 300), minval=0, maxval=1834, dtype=tf.int32)
    char_input = tf.random.uniform(shape=(batch_size, 300, 20), minval=0, maxval=132, dtype=tf.int32)
    pos_input = tf.random.uniform(shape=(batch_size, 300, 4), dtype=tf.float32)
    par_input = tf.random.uniform(shape=(batch_size, 300, 3), dtype=tf.float32)
    return [word_input, char_input, pos_input, par_input]

