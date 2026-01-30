from tensorflow.keras import layers
from tensorflow.keras import models

py
import sys
import numpy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

print(sys.version_info)


def tokenize_words(text_input):
    # Lowercase everything to standardize it
    text_input = text_input.lower()

    # Instantiate the tokenizer
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text_input)

    # If the create token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words("english"), tokens)
    return " ".join(filtered)


input_file = open("yejibro_data.txt").read()
# input_file = open("84-0.txt", "r", encoding="utf-8").read()

# Preprocces the input data, make tokens
processed_inputs = tokenize_words(input_file)

chars = sorted(set(processed_inputs))
char_to_num = dict((c, i) for i, c in enumerate(chars))

input_len = len(processed_inputs)
vocab_len = len(chars)
print(f"Total number of characters: {input_len}")
print(f"Total vocab: {vocab_len}")

seq_length = 100
x_data = []
y_data = []

# Loop through inputs, start at the beginning and go until we hit the final character we can create
# a sequence out of
for i in range(0, input_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]

    # Out sequence is the initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]

    # We now convert list of characters to integers based on previous mappings and add the values to
    # our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)
print(f"Total patterns: {n_patterns}")

x = numpy.reshape(x_data, (n_patterns, seq_length, 1))
x = x/float(vocab_len)

y = np_utils.to_categorical(y_data)

model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")

# print(model.summary())

filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
desired_callbacks = [checkpoint]

model.fit(x, y, epochs=8, batch_size=256, callbacks=desired_callbacks)