import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.preprocessing.text import tokenizer_from_json

import json

import numpy as np

import unicodedata
import re
import os
import io
import time
from collections import Counter

MAX_SEQUENCE_LENGTH = 30

def math_tokenizer(expression):
    regex = r'(\d+|[\+\-\*\/\(\)\^]|[a-zA-Z]+|.)'

    # Use the regex to split the expression into tokens
    _tok = re.findall(regex, expression)

    # Split numbers into individual digits
    ret_tokens = []
    for token in _tok:
        if token.isdigit():
            ret_tokens.extend(list(token))
        else:
            ret_tokens.append(token)

    # Filter out empty strings
    ret_tokens = [token for token in ret_tokens if token.strip()]

    return ret_tokens

def change_variable(expression):
    # Define regular expressions to match different tokens
    tokenstream = math_tokenizer(expression)
    variable = None
    for idx, tok in enumerate(tokenstream):
        if len(tok) == 1 and tok.isalpha():
          variable = tok
          tokenstream[idx] = "var"
    return ' '.join(tokenstream), variable

def text_cleaning(x):
  modified_text = [None] * len(x)
  tokens = set()
  tok_list = []

  variables = []

  for idx, dx in enumerate(x):
      lhs, rhs = dx.split("=")
      tokenstream, v = change_variable('='.join([lhs[2:-4], rhs[:-1]]))
      tokens.update(tokenstream)
      tok_list.extend(tokenstream)
      variables.append(v)
      modified_text[idx] = ''.join(tokenstream)

  return modified_text, variables

def generate_train_test_dataset(data, TRAIN_SIZE):
  modified_text, variables = text_cleaning(data)
  inputs = []
  targets = []

  for idx, dx in enumerate(modified_text):
      inp, tgt = dx.split("=")
      inputs.append(inp)
      targets.append(tgt)

  train_inputs = inputs[:TRAIN_SIZE]
  train_targets = targets[:TRAIN_SIZE]
  train_variables = variables[:TRAIN_SIZE]

  test_inputs = inputs[TRAIN_SIZE:]
  test_targets = targets[TRAIN_SIZE:]
  test_variables = variables[TRAIN_SIZE:]

  return train_inputs, train_targets, train_variables, test_inputs, test_targets, test_variables

  class MyDataset:
    def __init__(self, problem_type='calculus'):
        self.problem_type = 'calculus'
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None


    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, w):
        w = w.strip()

        w = 'start ' + w + ' end'
        return w

    def tokenize(self, lang, func):

        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=None, oov_token='<OOV>', analyzer=func)
        lang_tokenizer.fit_on_texts(lang)

        tensor = lang_tokenizer.texts_to_sequences(lang)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self, dataset, func):
        # creating cleaned input, output pairs
        targ_lang, inp_lang = dataset

        targ_lang = [self.preprocess_sentence(w) for w in targ_lang]
        inp_lang = [self.preprocess_sentence(w) for w in inp_lang]

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang, func)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang, func)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


    def call(self, dataset, BUFFER_SIZE, BATCH_SIZE, func):
        input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset(dataset, func)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')


  def call(self, x, hidden):
    x = self.embedding(x)
    output, h, c = self.lstm_layer(x, initial_state = hidden)
    return output, h, c

  def initialize_hidden_state(self):
    return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, max_length_input, max_length_output, attention_type='luong'):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type
    self.max_length_output = max_length_output

    # Embedding Layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Define the fundamental cell for decoder recurrent structure
    self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)



    # Sampler
    self.sampler = tfa.seq2seq.sampler.TrainingSampler()

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units,
                                                              None, self.batch_sz*[max_length_input], self.attention_type)

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = self.build_rnn_cell(batch_sz)

    # Define the decoder with respect to fundamental rnn cell
    self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)


  def build_rnn_cell(self, batch_sz):
    rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell,
                                  self.attention_mechanism, attention_layer_size=self.dec_units)
    return rnn_cell

  def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
    # ------------- #
    # typ: Which sort of attention (Bahdanau, Luong)
    # dec_units: final dimension of attention outputs
    # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
    # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

    if(attention_type=='bahdanau'):
      return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    else:
      return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state


  def call(self, inputs, initial_state):
    x = self.embedding(inputs)
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[self.max_length_output-1])
    return outputs

def loss_function(real, pred):
  # real shape = (BATCH_SIZE, max_length_output)
  # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss = cross_entropy(y_true=real, y_pred=pred)
  mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
  mask = tf.cast(mask, dtype=loss.dtype)
  loss = mask* loss
  loss = tf.reduce_mean(loss)
  return loss

import json
functions = '8exp^(9e)'

f = open('modified_train.txt', 'r')
modified_text = f.readlines()
f.close()

modified_text = [m[:-1] for m in modified_text]

BUFFER_SIZE = 32000
BATCH_SIZE = 64
# Let's limit the #training examples for faster training
num_examples = 30000

inputs = []
targets = []

N_TRAIN = 800000

for dx in (modified_text):
    try:
      inp, tgt = dx.split("=")
      inputs.append(inp)
      targets.append(tgt)
    except:
      print(f"Error at: {dx}")

train_inputs = inputs[:N_TRAIN]
train_targets = targets[:N_TRAIN]

test_inputs = inputs[N_TRAIN:]
test_targets = targets[N_TRAIN:]

data = (train_targets, train_inputs)

print("Creating the dataset")
dataset_creator = MyDataset('calculus')

# print("Training the tokenizer")
# train_dataset, inp_lang, targ_lang = dataset_creator.call(data, BUFFER_SIZE, BATCH_SIZE, math_tokenizer)

f = open('./inp_lang_tokenizer.json')
inp_json = f.read()
inp_json = json.loads(inp_json)
f.close()

f = open('./targ_lang_tokenizer.json')
targ_json = f.read()
targ_json = json.loads(targ_json)

inp_lang = tokenizer_from_json(inp_json)
targ_lang = tokenizer_from_json(targ_json)


# example_input_batch, example_target_batch = next(iter(train_dataset))
# example_input_batch.shape, example_target_batch.shape
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_length_input = 31
max_length_output = 31

embedding_dim = 128
units = 256
steps_per_epoch = num_examples//BATCH_SIZE

train_dataset, inp_lang, targ_lang = dataset_creator.call(data, BUFFER_SIZE, BATCH_SIZE, math_tokenizer)

## Test Encoder Stack
example_input_batch, example_target_batch = next(iter(train_dataset))
print("Creating encoder")
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)

sample_hidden = encoder.initialize_hidden_state()

# Test decoder stack

print("Creating decoder")
# vocab_size, embedding_dim, dec_units, batch_sz, max_length_input, max_length_output, attention_type='luong'
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, max_length_input, max_length_output, 'luong')
sample_x = tf.random.uniform((BATCH_SIZE, max_length_output))
decoder.attention_mechanism.setup_memory(sample_output)
initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)

print("Creating the optimizer")
optimizer = tf.keras.optimizers.Adam()

checkpoint_enc_dir = './training_checkpoints/encoder'
checkpoint_enc_prefix = os.path.join(checkpoint_enc_dir, "ckpt")

checkpoint_dec_dir = './training_checkpoints/decoder'
checkpoint_dec_prefix = os.path.join(checkpoint_dec_dir, "ckpt")

checkpoint_enc = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder)
checkpoint_dec = tf.train.Checkpoint(optimizer=optimizer,
                                 decoder=decoder)

# restoring the latest checkpoint in checkpoint_dir
checkpoint_enc.restore(tf.train.latest_checkpoint(checkpoint_enc_dir))
checkpoint_dec.restore(tf.train.latest_checkpoint(checkpoint_dec_dir))

print(decoder.embedding.variables)

[]