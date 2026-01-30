from tensorflow import keras

py
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers

print(tf.__version__)

# Model constants.
embedding_dim = 128
CHAR_DICT = 'abcdefghijklmnopqrstuvwxyz0123456789 .!?:,\'%-\(\)/$|&;[]"'


@tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
@tf.function
def split_text(text_list: list):
    joined = tf.strings.reduce_join(text_list)
    split = tf.strings.split(joined, sep="")
    return tf.expand_dims(split, 0)


def vectorize_layers(factor, time_dims):
    embedding_dim = 128
    text_input = layers.Input(shape=(1,), dtype=tf.string)
    vectorize_layer = TextVectorization(
        split=split_text,
        max_tokens=len(CHAR_DICT) + 1,
        output_mode='int',
        output_sequence_length=128)

    vectorize_layer.set_vocabulary(list(CHAR_DICT))
    x = vectorize_layer(text_input)
    x = layers.Embedding(len(CHAR_DICT) + 1, embedding_dim)(x)
    x = layers.Dense(factor, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    model = tf.keras.Model(inputs=text_input, outputs=x)
    return model


def build_model(input_labels, factor):
    inputs = []
    mergers = []
    for label in input_labels:
        text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name=label)
        inputs.append(text_input)
        mergers.append(text_input)

    x = layers.Concatenate()(mergers)
    x = tf.expand_dims(x, axis=-1)
    x = layers.TimeDistributed(vectorize_layers(factor, len(input_labels)))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(factor)(x)
    x = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


model = build_model(["a", "b"], 10)
# model.save("/tmp/asd")

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit({"a": np.array(["heasdasdre", "is somdfas", "dfae ", "mads d!", "asd dfa%# 12"]),
           "b": np.array(["asagda gdaasdfasd fgaewgasdg", "asd", "asdfasd,%#", "dasd", "asdasfg"])},
          np.array([0.1, 0.5, 0.4, 0.6, 0.3]))

print(model.predict({"a": ["here is some mad dog shit right ere boiz!"], "b": ["asdagaewgasdg"]}))

py

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers

print(tf.__version__)

# Model constants.
embedding_dim = 128
CHAR_DICT = 'abcdefghijklmnopqrstuvwxyz0123456789 .!?:,\'%-\(\)/$|&;[]"'


@tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
@tf.function
def split_text(text_list: list):
    joined = tf.strings.reduce_join(text_list)
    split = tf.strings.split(joined, sep="")
    return tf.expand_dims(split, 0)


def vectorize_layers(factor, time_dims):
    embedding_dim = 128
    text_input = layers.Input(shape=(1))
    # text_input = layers.Input(shape=(1,), dtype=tf.string)
    # vectorize_layer = TextVectorization(
    #     split=split_text,
    #     max_tokens=len(CHAR_DICT) + 1,
    #     output_mode='int',
    #     output_sequence_length=128)

    # vectorize_layer.set_vocabulary(list(CHAR_DICT))
    # x = vectorize_layer(text_input)
    x = layers.Embedding(len(CHAR_DICT) + 1, embedding_dim)(text_input)
    x = layers.Dense(factor, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    model = tf.keras.Model(inputs=text_input, outputs=x)
    return model


def build_model(input_labels, factor):
    inputs = []
    mergers = []
    for label in input_labels:
        text_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name=label)
        inputs.append(text_input)
        mergers.append(text_input)

    x = layers.Concatenate()(mergers)
    x = tf.expand_dims(x, axis=-1)
    x = layers.TimeDistributed(vectorize_layers(factor, len(input_labels)))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(factor)(x)
    x = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


model = build_model(["a", "b"], 10)
# model.save("/tmp/asd")

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit({"a": np.array([4, 5, 6, 6, 7], dtype=np.float),
           "b": np.array([6, 4, 2, 1, 0], dtype=np.float)},
          np.array([0.1, 0.5, 0.4, 0.6, 0.3], dtype=np.float))

print(model.predict({"a": np.array([2.5], dtype=np.float), "b": np.array([2.5], dtype=np.float)}))

py
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers

print(tf.__version__)

# Model constants.
embedding_dim = 128
CHAR_DICT = 'abcdefghijklmnopqrstuvwxyz0123456789 .!?:,\'%-\(\)/$|&;[]"'


@tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
@tf.function
def split_text(text_list: list):
    joined = tf.strings.reduce_join(text_list)
    split = tf.strings.split(joined, sep="")
    return tf.expand_dims(split, 0)


def build_model(factor, time_dims):
    embedding_dim = 128
    text_input = layers.Input(shape=(1,), dtype=tf.string)
    vectorize_layer = TextVectorization(
        split=split_text,
        max_tokens=len(CHAR_DICT) + 1,
        output_mode='int',
        output_sequence_length=128)

    vectorize_layer.set_vocabulary(list(CHAR_DICT))
    x = vectorize_layer(text_input)
    x = layers.Embedding(len(CHAR_DICT) + 1, embedding_dim)(x)
    x = layers.Dense(factor, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    model = tf.keras.Model(inputs=text_input, outputs=x)
    return model


model = build_model(10, 10)
# model.save("/tmp/asd")

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit( np.array(["heasdasdre", "is somdfas", "dfae ", "mads d!", "asd dfa%# 12"]),
          np.array([0.1, 0.5, 0.4, 0.6, 0.3]))

print(model.predict(
    np.array(["asdf gadg asdfas df hr hrt"]),))

CHAR_DICT = 'abcdefghijklmnopqrstuvwxyz0123456789 .!?:,\'%-\(\)/$|&;[]"'

CHAR_DICT = 'abcdefghijklmnopqrstuvwxyz0123456789 .!?:,\'%-()/$|&;[]"'