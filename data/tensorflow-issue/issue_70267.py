from tensorflow.keras import layers
from tensorflow.keras import models

import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

nltk.download('punkt')

corpus = [
    "Hello, how are you?",
    "I am fine, thank you!",
    "What are you doing today?",
    "I am working on a project."
]

# Tokenize the text
corpus_tokens = [word_tokenize(doc.lower()) for doc in corpus]

# Flatten the list of token lists into a single list of tokens
all_tokens = [token for sublist in corpus_tokens for token in sublist]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_tokens)
total_words = len(tokenizer.word_index) + 1

# Convert text to sequences of tokens
input_sequences = []
for line in corpus_tokens:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
label = to_categorical(label, num_classes=total_words)

# Build model
def build_model(max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len - 1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model(max_sequence_len, total_words)

# Training model
model.fit(predictors, label, epochs=1, verbose=1)  # Reduced epochs for demonstration

# Predict function
def predict_next_word(model, tokenizer, text, k):
    sequence = word_tokenize(text.lower())
    token_list = tokenizer.texts_to_sequences([sequence])[0]
    token_list = pad_sequences([token_list], maxlen=k, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == np.argmax(predicted):
            output_word = word
            break
    return output_word

print(predict_next_word(model, tokenizer, "I am", k=2))

import tensorflow as tf
print(tf.__version__)