# tf.random.uniform((B, 256, 256, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, Flatten, RepeatVector,
    Embedding, LSTM, Input, concatenate
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# This model is adapted and merged from the original Keras Sequential image model + functional language model + decoder.
# Fixed the data generator yield type to proper (inputs, labels) tuple to avoid gradient problems.
# Assumptions/notes:
# - Input image shape: (256, 256, 3)
# - Language input length max_length = 48 tokens
# - Vocabulary size from tokenizer + 1 for padding
# - Model outputs softmax vocabulary distribution for next token in sequence

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, max_length):
        super().__init__()
        # Image encoder (Sequential CNN + Dense + RepeatVector)
        self.image_model = Sequential([
            Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)),
            Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.3),
            Dense(1024, activation='relu'),
            Dropout(0.3),
            RepeatVector(max_length)  # Repeat vector to match language sequence length
        ])

        # Language encoder (Embedding + 2 stacked LSTMs)
        self.embedding = Embedding(vocab_size, 50, input_length=max_length, mask_zero=True)
        self.lstm1 = LSTM(128, return_sequences=True)
        self.lstm2 = LSTM(128, return_sequences=True)

        # Decoder (concatenate image and language encodings + 2 LSTMs + final Dense softmax)
        self.concat = concatenate
        self.decoder_lstm1 = LSTM(512, return_sequences=True)
        self.decoder_lstm2 = LSTM(512, return_sequences=False)
        self.dense_output = Dense(vocab_size, activation='softmax')

    def call(self, inputs, training=False):
        # inputs is expected to be a list/tuple: [image_batch, language_batch]
        images, language_tokens = inputs

        # Pass images through CNN image encoder
        image_features = self.image_model(images, training=training)  # shape (B, max_length, 1024)

        # Pass language through embedding and LSTMs
        embedded = self.embedding(language_tokens)
        x = self.lstm1(embedded)
        x = self.lstm2(x)

        # Concatenate image and language features
        merged = tf.concat([image_features, x], axis=-1)

        # Decode sequence to vocab prediction
        x = self.decoder_lstm1(merged)
        x = self.decoder_lstm2(x)
        output = self.dense_output(x)  # shape (B, vocab_size)

        return output


def my_model_function():
    # For demonstration purpose, we pick vocab_size=10000 and max_length=48 (matching the code)
    # Replace these with actual from your tokenizer
    vocab_size = 10000  # placeholder, replace with tokenizer vocab size +1
    max_length = 48     # max sequence length used in model

    model = MyModel(vocab_size, max_length)

    # Compile model with RMSprop optimizer and categorical crossentropy loss as original code
    optimizer = RMSprop(learning_rate=0.0001, clipvalue=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

def GetInput():
    # Return a tuple of tensors matching expected input:
    # images: shape (B, 256, 256, 3), float32, normalized [0,1]
    # language tokens: shape (B, 48), int32, token indexes 0 to vocab_size-1
    B = 2  # arbitrary batch size
    H, W, C = 256, 256, 3
    max_length = 48
    vocab_size = 10000  # must match model vocab size

    # Random float images
    images = tf.random.uniform((B, H, W, C), dtype=tf.float32)

    # Random integer token sequences in the valid vocab range
    language_tokens = tf.random.uniform(
        (B, max_length),
        minval=1, maxval=vocab_size, dtype=tf.int32
    )
    return [images, language_tokens]

