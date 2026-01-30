from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import os

# Set the TensorFlow logging level to ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Training data
input_data = [
    "Привет, как тебя зовут?",
    "Какие у тебя интересы?",
    "Как прошел твой день?",
    "Ты любишь путешествовать?",
    "Что ты думаешь о знакомствах через интернет?"
]

output_data = [
    "Привет! Меня зовут ЧатБот. А тебя?",
    "Мои интересы - это общение с людьми!",
    "Мой день прошел хорошо, спасибо.",
    "Я бот, поэтому путешествовать не могу, но обожаю общение с людьми!",
    "Я думаю, что знакомства через интернет - это отличный способ найти новых друзей и партнеров."
]

input_tokens = tf.keras.preprocessing.text.Tokenizer(filters='')
output_tokens = tf.keras.preprocessing.text.Tokenizer(filters='')
input_tokens.fit_on_texts(input_data)
output_tokens.fit_on_texts(output_data)

input_sequences = input_tokens.texts_to_sequences(input_data)
output_sequences = output_tokens.texts_to_sequences(output_data)

# Pad sequences to the maximum length
max_seq_length = max(len(seq) for seq in input_sequences + output_sequences)
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, padding='post', maxlen=max_seq_length)
output_sequences = tf.keras.preprocessing.sequence.pad_sequences(output_sequences, padding='post', maxlen=max_seq_length)

# Seq2Seq Model
latent_dim = 64

# Encoder
encoder_inputs = Input(shape=(max_seq_length,))
encoder_embedding = tf.keras.layers.Embedding(len(input_tokens.word_index) + 1, latent_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_seq_length,))
decoder_embedding = tf.keras.layers.Embedding(len(output_tokens.word_index) + 1, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(output_tokens.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# Prepare target sequences
target_sequences = output_sequences[:, 1:]

# Training
model.fit([input_sequences, output_sequences[:, :-1]], target_sequences,  epochs=50, batch_size=1)





# Save the model
model.save("chatbot_model.keras")

# Example of using the trained model
def predict_response(input_text):
    input_seq = input_tokens.texts_to_sequences([input_text])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='post', maxlen=max_seq_length)
    output_seq = model.predict([input_seq, np.zeros((len(input_seq), max_seq_length))])
    output_seq = np.argmax(output_seq, axis=-1)
    output_text = ' '.join([output_tokens.index_word[i] for i in output_seq[0] if i != 0])
    return output_text

# Example of using the model to generate responses
user_input = "Как прошел твой день?"
response = predict_response(user_input)
print(response)