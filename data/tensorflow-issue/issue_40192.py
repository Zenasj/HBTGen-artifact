from tensorflow.keras import layers

model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 2, input_length = max_len))
model.add(keras.layers.Flatten())             
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

import numpy as np
import tensorflow.keras as keras

from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Define docs
docs = [
    'Amazing!',
    'Well done!',
    'Great job!',
    'Good job',
    'Nicely done!',
    'Amazing job!',
    'Not bad!',
    'Very poor.',
    'Not well',
    'Not very well written',
    'Purely bad',
    'It\'s not good.',
    'Very badly done',
    'Very bad'
]
    
# Define labels
labels = np.array([1] * 7 + [0] * 7)

# Tokenize the docs
vocab_size = 50

tokenizer = keras.preprocessing.text.Tokenizer(vocab_size, oov_token = '<UNK>')
tokenizer.fit_on_texts(docs)

seqs = tokenizer.texts_to_sequences(docs)

# Pad sequences
max_len = max([len(x) for x in seqs]) + 2

padded_seqs = pad_sequences(seqs, 
                            maxlen  = max_len, 
                            padding = 'post')

model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 32, input_length = max_len))
model.add(keras.layers.Flatten())              # We need to flatten the array before the dense layer
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

# Compile the model
model.compile(optimizer = 'adam', 
              loss      = 'binary_crossentropy', 
              metrics   = ['accuracy'])

# Fit the model
model.fit(padded_seqs, labels, epochs = 150, verbose = 10)

# Note that after all the model fits the data and loss goes down as expected:
# plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['loss'])