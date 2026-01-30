from tensorflow import keras
from tensorflow.keras import layers

restore = True # resp False
checkpoint_dir = "./ckpt-py/"
restore_from = "./ckpt-py/-1"

checkpoint = tfe.Checkpoint(optimizer=optimizer,
                            encoder = encoder,
                            decoder = decoder,
                            optimizer_step=tf.train.get_or_create_global_step())

if restore == False:
  for epoch in range(EPOCHS):
      ...
      save_path = checkpoint.save(checkpoint_dir)
      metadata = tf.contrib.checkpoint.object_metadata(save_path)
      with open("save_python.txt", "w") as f: 
        f.write(str(metadata))
else: 
  rst = checkpoint.restore(restore_from)
  # or 
  #rst = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  metadata = tf.contrib.checkpoint.object_metadata(restore_from)
  with open("restore_python.txt", "w") as f: 
    f.write(str(metadata))
  #print(rst.assert_consumed())

rst.assert_consumed()

from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf
tfe = tf.contrib.eager

tf.enable_eager_execution()

from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time

print(tf.__version__)

# Download the file
#path_to_zip = tf.keras.utils.get_file(
#    'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip', 
#    extract=True)

path_to_file = "data/spa.txt"


# Add a start and end token to each sentence.
# Clean the sentences by removing special characters.
# Create a word index and reverse word index (dictionaries mapping from word -> id and id -> word).
# Pad each sentence to a maximum length.


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    
    return word_pairs

    # This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
  def __init__(self, lang):
    self.lang = lang
    self.word2idx = {}
    self.idx2word = {}
    self.vocab = set()
    
    self.create_index()
    
  def create_index(self):
    for phrase in self.lang:
      self.vocab.update(phrase.split(' '))
    
    self.vocab = sorted(self.vocab)
    
    self.word2idx['<pad>'] = 0
    for index, word in enumerate(self.vocab):
      self.word2idx[word] = index + 1
    
    for word, index in self.word2idx.items():
      self.idx2word[index] = word

def max_length(tensor):
    return max(len(t) for t in tensor)

def load_dataset(path, num_examples):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)
    # index language using the class defined above    
    inp_lang = LanguageIndex(sp for en, sp in pairs)
    targ_lang = LanguageIndex(en for en, sp in pairs)
    # Vectorize the input and target languages
    # Spanish sentences
    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]
    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar

num_examples = 100
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file, num_examples)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))

# The input is put through an encoder model which gives us the encoder output of shape 
# (batch_size, max_length, hidden_size) and the encoder hidden state of shape (batch_size, hidden_size). 

# We're using Bahdanau attention. Lets decide on notation before writing the simplified form:
# 
#     FC = Fully connected (dense) layer
#     EO = Encoder output
#     H = hidden state
#     X = input to the decoder
# 
# And the pseudo-code:
# 
#     score = FC(tanh(FC(EO) + FC(H)))
#     attention weights = softmax(score, axis = 1). 
#          Softmax by default is applied on the last axis but here we want to apply it on the 1st axis,
#          since the shape of score is (batch_size, max_length, hidden_size). 
#          Max_length is the length of our input. Since we are trying to assign a weight to each input,
#          softmax should be applied on that axis.
#     context vector = sum(attention weights * EO, axis = 1). Same reason as above for choosing axis as 1.
#     embedding output = The input to the decoder X is passed through an embedding layer.
#     merged vector = concat(embedding output, context vector)
#     This merged vector is then given to the GRU

def gru(units):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
  if tf.test.is_gpu_available():
    return tf.keras.layers.CuDNNGRU(units, 
                                    return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform')
  else:
    return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)        
        return output, state
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # score shape == (batch_size, max_length, hidden_size)
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * max_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size * max_length, vocab)
        x = self.fc(output)
        return x, state, attention_weights
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

restore = True
checkpoint_dir = "./ckpt-py/"
restore_from = "./ckpt-py/-1"

checkpoint = tfe.Checkpoint(optimizer=optimizer,
                            encoder = encoder,
                            decoder = decoder,
                            optimizer_step=tf.train.get_or_create_global_step())


def loss_function(real, pred):
  mask = 1 - np.equal(real, 0)
  loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
  return tf.reduce_mean(loss_)


# The evaluate function is similar to the training loop, except we don't use teacher forcing here. The input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output.
# Stop predicting when the model predicts the end token.
# And store the attention weights for every time step.

def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        # storing the attention weigths to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        predicted_id = tf.multinomial(tf.exp(predictions), num_samples=1)[0][0].numpy()
        result += targ_lang.idx2word[predicted_id] + ' '
        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()
    
def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))
    #attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    #plot_attention(attention_plot, sentence.split(' '), result.split(' '))


#translate('hace mucho frio aqui.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

# Training
# 
# Pass the input through the encoder which return encoder output and the encoder hidden state.
# The encoder output, encoder hidden state and the decoder input (which is the start token) is passed to the decoder.
# The decoder returns the predictions and the decoder hidden state.
# The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
# Use teacher forcing to decide the next input to the decoder.
# Teacher forcing is the technique where the target word is passed as the next input to the decoder.
# The final step is to calculate the gradients and apply it to the optimizer and backpropagate.


EPOCHS = 5

if restore == False:
  for epoch in range(EPOCHS):
      start = time.time()
      hidden = encoder.initialize_hidden_state()
      total_loss = 0
      for (batch, (inp, targ)) in enumerate(dataset):
          loss = 0
          with tf.GradientTape() as tape:
              enc_output, enc_hidden = encoder(inp, hidden)
              dec_hidden = enc_hidden
              dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)       
              # Teacher forcing - feeding the target as the next input
              for t in range(1, targ.shape[1]):
                  # passing enc_output to the decoder
                  predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                  loss += loss_function(targ[:, t], predictions)
                  # using teacher forcing
                  dec_input = tf.expand_dims(targ[:, t], 1)
          total_loss += (loss / int(targ.shape[1]))
          variables = encoder.variables + decoder.variables
          gradients = tape.gradient(loss, variables)
          optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
          #if batch % 100 == 0:
          #    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
          #                                                 batch,
          #                                                 loss.numpy() / int(targ.shape[1])))
      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss/len(input_tensor)))
      #print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
      save_path = checkpoint.save(checkpoint_dir)
      print(save_path)
      metadata = tf.contrib.checkpoint.object_metadata(save_path)
      with open("save_python.txt", "w") as f: 
        f.write(str(metadata))
      translate('Sean gentiles.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
      translate('Escuche.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
      translate('Estoy trabajando.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
else: 
  rst = checkpoint.restore(restore_from)
  #rst = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  metadata = tf.contrib.checkpoint.object_metadata(restore_from)
  with open("restore_python.txt", "w") as f: 
    f.write(str(metadata))
  print(rst.assert_consumed())
  translate('Sean gentiles.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
  translate('Escuche.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
  translate('Estoy trabajando.', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

"./ckpt-py/-5"

rst = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))