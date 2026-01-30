from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

class LuongAttention(tf.keras.Model):
    def __init__(self, input_dim=max_len, att_type='dot'):
        super(LuongAttention, self).__init__()
        w_init = tf.random_normal_initializer()
        self.att_type = att_type
        self.W = tf.Variable(
            initial_value = np.identity(input_dim, dtype='float32'),
            trainable=False
        )
        self.WLayer = tf.keras.layers.Dense(input_dim, 
                                           kernel_regularizer=tf.keras.regularizers.l2(0.01))
        if self.att_type == 'general':
            self.W = tf.Variable(
                initial_value=w_init(shape=(input_dim, input_dim), dtype="float32"),
                trainable=True
            )
        self.loss = 1e-5 * tf.nn.l2_loss(self.W)
    def call(self, inputs):
        self.add_loss(self.loss)
        if self.att_type == 'concat':
            score = tf.matmul(inputs[0], self.WLayer(inputs[1]), transpose_b=True)
            alignment = tf.nn.softmax(score, axis=2)
        else:
            score = tf.matmul(self.W, inputs[1])
            score = tf.matmul(inputs[0], score, transpose_b=True)
            alignment = tf.nn.softmax(score, axis=2)
        return alignment

# Build the model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Attention
from tensorflow.keras.layers import Dense, Concatenate, Reshape
from tensorflow.keras.activations import linear
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import Model

in1 = Input((max_len,))
in2 = Input((max_len,))

emb = Embedding(len(tokenizer.word_index) + 1, 
                embedding_dim, 
                input_length=max_len)
if use_embedding: 
    emb = Embedding(len(tokenizer.word_index) + 1, 
                    embedding_dim, 
                    weights=[embedding_matrix],
                    input_length=max_len, 
                    trainable=emb_trainable)

# First sequence branch
branch1 = emb(in1)
branch1l = Bidirectional(LSTM(25, 
                              activation='elu', 
                              return_sequences=True, 
                              kernel_regularizer=l2(0.01)))(branch1)

# Second sequence branch
branch2 = emb(in2)
branch2l = Bidirectional(LSTM(25, 
                              activation='elu', 
                              return_sequences=True, 
                              kernel_regularizer=l2(0.01)))(branch2)

# Concatenate with Luong attention (eventually add word embedding as value)
attention1 = LuongAttention(input_dim=max_len, 
                            att_type='general')([branch1, branch2])
attention2 = LuongAttention(input_dim=max_len, 
                            att_type='general')([branch1l, branch2l])
concat = Concatenate(axis=-1)([attention1, attention2])

# Process concatenated output
out = Bidirectional(LSTM(80, activation='elu', kernel_regularizer=l2(0.01)))(concat)
out = Dense(1, activation='sigmoid')(out)

model = Model(inputs=[in1, in2], outputs=[out])
model.compile(loss='mean_squared_error', 
              optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

# Fit the model
from tensorflow.keras.callbacks import ReduceLROnPlateau

my_callbacks = [
    ReduceLROnPlateau(factor=0.25, mode="min")
]

history2 = model.fit([X_train[:, 0, :], X_train[:, 1, :]],
                    y_train, 
                    epochs=epochs, 
                    validation_split=0.2,
                    use_multiprocessing=True,
                    sample_weight=sample_weights,
                    callbacks=my_callbacks)

import csv
import pandas as pd
import numpy as np

# read in sentence dataset
with open('SICK.txt', 'r') as file:
    sdf = pd.read_table(file)

# re-edit scores so that it's a single relatedness measure
scoreAB = []
scoreBA = []
for pair in sdf.iterrows():
    if pair[1]['entailment_AB'] == 'A_entails_B':
        scoreAB.append(pair[1]['relatedness_score'] - 1)
    elif pair[1]['entailment_AB'] == 'A_neutral_B':
        scoreAB.append(pair[1]['relatedness_score'] - 3)
    else:
        scoreAB.append(1 - pair[1]['relatedness_score'])
    if pair[1]['entailment_BA'] == 'B_entails_A':
        scoreBA.append(pair[1]['relatedness_score'] - 1)
    elif pair[1]['entailment_BA'] == 'B_neutral_A':
        scoreBA.append(pair[1]['relatedness_score'] - 3)
    else:
        scoreBA.append(1 - pair[1]['relatedness_score'])
sdf['relatedness_score_AB'] = scoreAB
sdf['relatedness_score_BA'] = scoreBA

# define hyperparameters
vocab_size = None
oov_tok = '<OOV>'
max_len = 75
pad_pos = 'post'
trunc_pos = 'post'
embedding_dim = 50
epochs = 25
emb_trainable = False
use_embedding = True
dim_factor = 0.7
out_range = (0, 1)

# A little bit of preprocessing

# First, add beginning and end of string words
trim = lambda x: x #"<BOS> " + x + " <EOS>" if "<BOS>" not in x else x

sdf['sentence_A'] = sdf['sentence_A'].apply(trim)
sdf['sentence_B'] = sdf['sentence_B'].apply(trim)
sdf['sentence_A_original'] = sdf['sentence_A_original'].apply(trim)
sdf['sentence_B_original'] = sdf['sentence_B_original'].apply(trim)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sdf['sentence_A'] + 
                       sdf['sentence_B'] + 
                       sdf['sentence_A_original'] + 
                       sdf['sentence_B_original'])
sa = tokenizer.texts_to_sequences(sdf['sentence_A'])
sb = tokenizer.texts_to_sequences(sdf['sentence_B'])
sao = tokenizer.texts_to_sequences(sdf['sentence_A_original'])
sbo = tokenizer.texts_to_sequences(sdf['sentence_B_original'])

sa = pad_sequences(sa, maxlen=max_len, padding=pad_pos, truncating=trunc_pos)
sb = pad_sequences(sb, maxlen=max_len, padding=pad_pos, truncating=trunc_pos)
sao = pad_sequences(sao, maxlen=max_len, padding=pad_pos, truncating=trunc_pos)
sbo = pad_sequences(sbo, maxlen=max_len, padding=pad_pos, truncating=trunc_pos)

# construct sentence dataset
ds_pairs = (sa, 
            sb, 
            sdf['relatedness_score_AB'])
orig_pairs = (sao, 
              sbo, 
              [dim_factor * x for x in sdf['relatedness_score_AB']])

# construct reversed semantic relationships
ds_pairs_reversed = (sb, 
                     sa, 
                     sdf['relatedness_score_BA'])
orig_pairs_reversed = (sbo,
                       sao,
                       [dim_factor * x for x in sdf['relatedness_score_BA']])

"""
uncomment to include reversed semantic relationships in dataset
"""
# ds_pairs = np.concatenate([ds_pairs, ds_pairs_reversed])
# orig_pairs = np.concatenate([orig_pairs, orig_pairs_reversed])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# set up
X_ds = np.transpose(np.array([ds_pairs[0], ds_pairs[1]]), axes=(1,0,2))
X_orig = np.transpose(np.array([orig_pairs[0], orig_pairs[1]]), axes=(1,0,2))
y_ds = np.array(ds_pairs[2])
y_orig = np.array(orig_pairs[2])

# normalize the output values for the sake of everything
scaler = MinMaxScaler(feature_range=out_range)
scaler.fit(np.concatenate([y_ds, y_orig]).reshape(-1, 1))
y_ds = scaler.transform(y_ds.reshape(-1, 1)).transpose()[0, :]
y_orig = scaler.transform(y_orig.reshape(-1, 1)).transpose()[0, :]

"""
Uncomment to incorporate original sentences into training set.
Not recommended, because semantic relationships aren't as strong.
Tune hyperparameter dim_factor to establish how much you want to edit score.
"""
# X = np.concatenate([X_ds, X_orig])
# y = np.concatenate([y_ds, y_orig])

# Split up 
X = X_ds
y = y_ds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# We define the embedding layer here.
import os

embeddings_index = {}
f = open(os.path.join('./glove', 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Build the model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Attention
from tensorflow.keras.layers import Dense, Concatenate, Reshape
from tensorflow.keras.activations import linear
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import Model

in1 = Input((max_len,))
in2 = Input((max_len,))

emb = Embedding(len(tokenizer.word_index) + 1, 
                embedding_dim, 
                input_length=max_len)
if use_embedding: 
    emb = Embedding(len(tokenizer.word_index) + 1, 
                    embedding_dim, 
                    weights=[embedding_matrix],
                    input_length=max_len, 
                    trainable=emb_trainable)

# First sequence branch
branch1 = emb(in1)
branch1l = Bidirectional(LSTM(25, 
                              activation='elu', 
                              return_sequences=True, 
                              kernel_regularizer=l2(0.01)))(branch1)

# Second sequence branch
branch2 = emb(in2)
branch2l = Bidirectional(LSTM(25, 
                              activation='elu', 
                              return_sequences=True, 
                              kernel_regularizer=l2(0.01)))(branch2)

# Concatenate with Luong attention (eventually add word embedding as value)
attention1 = Attention(use_scale=True)([branch1, branch2])
attention2 = Attention(use_scale=True)([branch1l, branch2l])
concat = Concatenate(axis=-1)([attention1, attention2])

# Process concatenated output
out = Bidirectional(LSTM(80, activation='elu', kernel_regularizer=l2(0.01)))(concat)
out = Dense(1, activation='sigmoid')(out)

model = Model(inputs=[in1, in2], outputs=[out])
model.compile(loss='mean_squared_error', 
              optimizer='adam', 
              metrics=['kullback_leibler_divergence'])
model.summary()

# Fit the model
from tensorflow.keras.callbacks import ReduceLROnPlateau

my_callbacks = [
    ReduceLROnPlateau(factor=0.25,
                      mode="min")
]

history1 = model.fit([X_train[:, 0, :], X_train[:, 1, :]],
                    y_train, 
                    epochs=epochs, 
                    validation_split=0.2,
                    callbacks=my_callbacks)

model.evaluate([X_test[:, 0, :], X_test[:, 1, :]], y_test, batch_size=25)