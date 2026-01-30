import math
import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

mc_pred = self.main_classifier([xu, xs], training=True)

tf-nighly

persistent=True

import string
import nltk
import time

nltk.download('punkt')

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import *

# Load dataset
example_df = pd.read_csv('example.csv')

# Sample train/validation/test data
np.random.seed(100)
train, validation, test = np.split(example_df.sample(frac=1), [int(.7*len(example_df)), int(.85*len(example_df))])

# Class Weight Function
def compute_sample_weights(df, target, t_expect, weight_name):
  # Setup mitigator_weight
  df[weight_name] = 0

  # Get frequencies per target level
  targets = df.groupby(target).size()

  # Compute sample weights
  target_weights = t_expect / (targets / targets.sum())

  # Convert to dictionary
  target_dict = target_weights.to_dict()

  # Add sample weights to dataframe
  for i in target_dict:
    df[weight_name] = np.where((df[target] == i), target_dict[i], df[weight_name])
  
  return df

# Compute Main Class Weights
train = compute_sample_weights(df=train, target='target', t_expect=(1/3), weight_name='mainClass_weight')

# Compute Protect Class Weights
train = compute_sample_weights(df=train, target='protect', t_expect=(1/2), weight_name='protectClass_weight')

# Preprocess Text Data
vocab_size = 25000
max_length = 300
padding_type = 'post'
trunc_type = 'post'
oov_tok = '<unk>'

def Text_to_Seq(train, val, test, vocab_size, max_length, padding_type, trunc_type, oov_tok):
    # Text tokenization
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train)

    # Create word_index
    word_index = tokenizer.word_index

    # Transforms each text doc to a sequence of integers in train, val and test
    x = tokenizer.texts_to_sequences(train)
    y = tokenizer.texts_to_sequences(val)
    z = tokenizer.texts_to_sequences(test)

    # Pad sequences to the same length
    x = pad_sequences(x, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    y = pad_sequences(y, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    z = pad_sequences(z, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return x, y, z, word_index

train_text = train['reviewText']
val_text = validation['reviewText']
test_text = test['reviewText']

train_text, val_text, test_text, word_index = Text_to_Seq(train_text, val_text, test_text, vocab_size, max_length, padding_type, trunc_type, oov_tok)

# Create input, target, protect, sample_weights for Train
xu_train = np.array(train_text, dtype=np.int32)
xs_train = np.array(train.iloc[:, np.r_[0, 2, 10:30]], dtype=np.int32)
y_train = np.array(train[['target_negative', 'target_neutral', 'target_positive']], dtype=np.float32)
z_train = np.array(train['protect_m'], dtype=np.float32).reshape((-1,1))
mainClass_weight = np.array(train['mainClass_weight'], dtype=np.float32).reshape((-1,1))
protectClass_weight = np.array(train['protectClass_weight'], dtype=np.float32).reshape((-1,1))

# Create input, target, protect for Validation
xu_val = np.array(val_text, dtype=np.int32)
xs_val = np.array(validation.iloc[:, np.r_[0, 2, 8:28]], dtype=np.int32)
y_val = np.array(validation[['target_negative', 'target_neutral', 'target_positive']], dtype=np.float32)
z_val = np.array(validation['protect_m'], dtype=np.float32).reshape((-1,1))

# Create input, target, protect for Test
xu_test = np.array(test_text, dtype=np.int32)
xs_test = np.array(test.iloc[:, np.r_[0, 2, 8:28]], dtype=np.int32)
y_test = np.array(test[['target_negative', 'target_neutral', 'target_positive']], dtype=np.float32)
z_test = np.array(test['protect_m'], dtype=np.float32).reshape((-1,1))

# Setup Pretrained GloVe Embedding
def load_embedding(file_path):
    # Initialize embeddings_index
    embeddings_index = {}

    # Store pretrained word vectors in embeddings_index
    with open(file_path) as f:
        for line in f:
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Print number of word vectors found
    print("Found %s word vectors." % len(embeddings_index))

    return embeddings_index

embeddings_index = load_embedding('glove.840B.300d.txt') # obtained from https://nlp.stanford.edu/projects/glove/

# Create Embedding matrix
num_tokens = min(vocab_size, len(word_index))+1
embed_dim = 300

def embedding_matrix(word_index, embeddings_index, num_tokens, embed_dim):
    # Initialize embedding_matrix and counters
    hits = 0
    misses = 0
    embedding_matrix = np.zeros((num_tokens, embed_dim))

    # Create embedding_matrix
    for word, i in word_index.items():
      if i > vocab_size:
        continue
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
      else:
        misses += 1

    # Print number of hits and misses
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix

embedding_matrix = embedding_matrix(word_index, embeddings_index, num_tokens, embed_dim)

class model_components:

  def mitigation_expert():
    inputs = Input(shape=(300,), dtype=tf.int32, name="me_input")
    x = Embedding(num_tokens, 300, weights=[embedding_matrix], input_length=max_length, trainable=False, name="me_embedding")(inputs)
    x = LSTM(300, return_sequences=False, name="me_lstm")(x)

    model = Model(inputs, x)

    return model

  def control_expert():
    inputs = Input(shape=(22,), dtype=tf.int32, name="ce_input")
    y = Dense(19, activation='relu', name="ce_hidden")(inputs)

    model = Model(inputs, y)

    return model

  def main_classifier():
    # Expert components
    me = model_components.mitigation_expert()
    ce = model_components.control_expert()

    # Main classifier
    ensemble = concatenate([me.output, ce.output], name="pred_ensemble")
    pred_output = Dense(319, activation="relu", name="pred_hidden")(ensemble)
    pred_output = Dense(3, activation="softmax", name="pred_output")(pred_output)

    model = Model(inputs=[me.input, ce.input], outputs=pred_output, name="main_classifier")

    return model
  
  def adversary_classifier():
    # Mitigation Expert component
    me = model_components.mitigation_expert()

    # Adversary classifier
    adv_output = Dense(300, activation='relu', name="adv_hidden")(me.output)
    adv_output = Dense(1, activation='sigmoid', name="adv_output")(adv_output)

    model = Model(inputs=me.input, outputs=adv_output, name="adversary_classifier")

    return model

def tf_normalize(x):
  return x / (tf.norm(x) + np.finfo(np.float32).tiny)

class DebiasModel(keras.Model):
    def __init__(self, main_classifier, adversary_classifier):
        super(DebiasModel, self).__init__()
        self.main_classifier = main_classifier
        self.adversary_classifier = adversary_classifier

    def compile(self, mc_optimizer, adv_optimizer, mc_loss, adv_loss, debias_param):
        super(DebiasModel, self).compile()
        self.mc_optimizer = mc_optimizer
        self.adv_optimizer = adv_optimizer
        self.mc_loss = mc_loss
        self.adv_loss = adv_loss
        self.debias_param = debias_param

    def train_step(self, data):
      # Unpack data from model.fit()
      x, y, sample_weight = data

      # Unpack input and output features
      xu, xs = x
      y_mc = y['pred_output']
      z_adv = y['adv_output']

      # Unpack sample_weights
      mainClass_weights = sample_weight["pred_output"]
      protectClass_weights = sample_weight["adv_output"]

      # Generate prediction and compute loss for Main_Classifier
      with tf.GradientTape() as mc_tape, tf.GradientTape() as me_mc_tape:
        mc_pred = self.main_classifier([xu, xs], training=True)
        mc_loss = self.mc_loss(y_mc, mc_pred, sample_weight=mainClass_weights)
      
      # Compute and Apply Gradients for CE & Main Classifier
      mc_trainable_vars = self.main_classifier.trainable_weights[3:]
      mc_grads = mc_tape.gradient(mc_loss, mc_trainable_vars)
      self.mc_optimizer.apply_gradients(zip(mc_grads, mc_trainable_vars))

      # Generate prediction and compute loss for Adversary_Classifier
      with tf.GradientTape() as adv_tape, tf.GradientTape() as me_adv_tape:
        adv_pred = self.adversary_classifier(xu)
        adv_loss = self.adv_loss(z_adv, adv_pred, sample_weight=protectClass_weights)
      
      # Compute and Apply Gradients for CE & Main Classifier
      adv_trainable_vars = self.adversary_classifier.trainable_weights[3:]
      adv_grads = adv_tape.gradient(adv_loss, adv_trainable_vars)
      self.adv_optimizer.apply_gradients(zip(adv_grads, adv_trainable_vars))

      # Compute and Apply Gradients to debias ME
      me_adv_debias_trainable_vars = self.adversary_classifier.trainable_weights[:3]
      adv_debias_grads = me_adv_tape.gradient(adv_loss, me_adv_debias_trainable_vars)
      adv_debias_dict = tf.lookup.StaticHashTable(
          tf.lookup.KeyValueTensorInitializer(me_adv_debias_trainable_vars, adv_debias_grads), 0)
      
      me_mc_debias_trainable_vars = self.main_classifier.trainable_weights[:3]
      mc_debias_grads = me_mc_tape.gradient(mc_loss, me_mc_debias_trainable_vars)

      me_grads = []

      for g, v in zip(mc_debias_grads, me_mc_debias_trainable_vars):
        unit_adv = tf_normalize(adv_debias_dict.lookup(v))
        g -= tf.math.reduce_sum(g * unit_adv) * unit_adv
        g -= self.debias_param * adv_debias_dict.lookup(v)
        me_grads.append(zip(g, v))
      
      self.mc_optimizer.apply_gradients(me_grads)
      
      return {"pred_loss": mc_loss, "adv_loss": adv_loss}

# Build and Fit Model
model = DebiasModel(model_components.main_classifier(),
                    model_components.adversary_classifier())

model.compile(mc_optimizer=tf.keras.optimizers.Adam(),
              adv_optimizer=tf.keras.optimizers.Adam(),
              mc_loss=tf.keras.losses.CategoricalCrossentropy(),
              adv_loss=tf.keras.losses.BinaryCrossentropy(),
              debias_param=1)

epoch = 5
sample_weights = {
    "pred_output": mainClass_weight,
    "adv_output": protectClass_weight,}

model.fit(x=[xu_train, xs_train],
          y={"pred_output": y_train, "adv_output": z_train},
          validation_data=([xu_val, xs_val], {"pred_output": y_val, "adv_output": z_val}),
          sample_weight=sample_weights,	epochs=epoch, batch_size=256, verbose=1)

fit()

train_step

with tf.GradientTape(persistent=True) as tape:
      # Forward pass
      pred = self([xu, xs], training=True) # Issue triggered here (see traceback)
      # Compute losses
      mc_loss = self.mainClass_loss(y_mc, pred[0], sample_weight=mainClass_weights)
      adv_loss = self.advClass_loss(z_adv, pred[1], sample_weight=protectClass_weights)

train_step