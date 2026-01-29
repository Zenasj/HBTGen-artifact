# tf.random.uniform((1, 75), dtype=tf.float32) ← Input shape is (batch_size=1, sequence_length=75), integer token IDs

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Lambda, Input
from tensorflow.keras.models import Model
import numpy as np

# Since the original model uses a CRF layer from an external library (likely keras_contrib or tfaddons),
# we will create a placeholder CRF-like layer that acts as an identity here. 
# For real use, replace this with proper CRF implementation compatible with TF 2.20.

class DummyCRF(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(DummyCRF, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # Normally, CRF has trainable transition params here
        self.trans_params = self.add_weight(name='trans_params', shape=(self.output_dim, self.output_dim),
                                            initializer='random_normal', trainable=True)
        super(DummyCRF, self).build(input_shape)

    def call(self, inputs, mask=None):
        # This dummy call just returns a linear projection plus softmax-like activation
        # In real CRF, decode Viterbi sequence
        # For compatibility with TFLite and compilation, keep it simple
        # Use a Dense layer to simulate output logits
        # Assuming inputs shape: (batch, seq_len, features)
        logits = tf.keras.layers.Dense(self.output_dim)(inputs)
        # For TFLite compatibility, output softmax probabilities (not decoded labels)
        outputs = tf.nn.softmax(logits, axis=-1)
        return outputs

    def get_config(self):
        config = super(DummyCRF, self).get_config()
        config.update({"output_dim": self.output_dim})
        return config

class MyModel(tf.keras.Model):
    def __init__(self, n_words=35000, max_len=75, embedding_dim=20, lstm_units=50, n_tags=20):
        super(MyModel, self).__init__()

        # Parameters from the shared issue:
        # n_words + 2 (PAD & UNK)
        self.n_words = n_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.n_tags = n_tags  # number of tags for NER - typically 20 
        # The original issue's param count stated embedding input_dim ~ n_words+2= 35000 approx.

        # Embedding layer
        self.embedding = Embedding(input_dim=self.n_words + 2,
                                   output_dim=self.embedding_dim, input_length=self.max_len,
                                   name='embedding')
        # Bidirectional LSTM
        self.bilstm = Bidirectional(LSTM(units=self.lstm_units, return_sequences=True, recurrent_dropout=0.1),
                                    name='bidirectional_lstm')

        # TimeDistributed Dense with relu
        self.time_distributed = TimeDistributed(Dense(50, activation="relu"), name='time_distributed_dense')

        # CRF layer placeholder
        # Original code uses: crf = CRF(n_tags + 1)
        self.crf = DummyCRF(self.n_tags + 1, name='crf')

        # Sequence mask layer that was provided in original code but not used explicitly in call
        self.sequence_mask = Lambda(lambda x: tf.greater(x, 0), name='sequence_mask')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # inputs: assumed shape (batch, max_len) with integer token IDs
        # Step 1: Embedding lookup
        x = self.embedding(inputs)  # shape: (batch, max_len, embedding_dim)

        # Step 2: BiLSTM
        x = self.bilstm(x, training=training)  # shape: (batch, max_len, lstm_units*2)

        # Step 3: TimeDistributed Dense with relu
        x = self.time_distributed(x)  # shape: (batch, max_len, 50)

        # Step 4: CRF - outputs softmax probabilities for each tag (dummy implementation)
        output = self.crf(x)  # shape: (batch, max_len, n_tags + 1)

        return output

def my_model_function():
    """
    Returns an instance of MyModel with default initialization.
    """
    # Common params inferred from the TF issue and usage:
    n_words = 35000  # inferred from embedding input dim ~703000/20=~35150; slightly rounded
    max_len = 75
    embedding_dim = 20
    lstm_units = 50
    n_tags = 20  # typical NER like CoNLL has 20 tags; original had 21 including padding probably

    return MyModel(n_words=n_words, max_len=max_len, embedding_dim=embedding_dim, lstm_units=lstm_units, n_tags=n_tags)

def GetInput():
    """
    Returns a random tensor input simulating batch input of integer token IDs, shape (1, 75)
    Tokens are integers between 0 and n_words+1 to match embedding input dim.
    """
    n_words = 35000
    max_len = 75
    # Use integers in [0, n_words+1] including PAD=0, UNK=1, etc.
    # Batch size = 1, sequence length = max_len
    random_tokens = tf.random.uniform(shape=(1, max_len), minval=0, maxval=n_words+2, dtype=tf.int32)
    return random_tokens

