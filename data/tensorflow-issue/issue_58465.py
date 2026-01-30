import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

def build_preprocessing_model(features):
    # Build the preprocessing model
    inputs = {}

    for name, column in features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    # print(inputs)
    
    # Concatenate numeric inputs
    numeric_inputs = {name: input for name, input in inputs.items() if input.dtype==tf.float32}
    # print(numeric_inputs)
    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = layers.Normalization()
    norm.adapt(np.array(features[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)
    # print(all_numeric_inputs)

    preprocessed_inputs = [all_numeric_inputs]

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue

        output_sequence_length = get_output_sequence_length(features[name])
        text_vectorizer = layers.TextVectorization(output_sequence_length=output_sequence_length)
        text_vectorizer.adapt(features[name])
        x = text_vectorizer(input)
        print(x.shape)

        x = tf.cast(x, tf.float32)
        preprocessed_inputs.append(x)
    # print(preprocessed_inputs)
    
    # for preprocessed_input in preprocessed_inputs:
    #     print(preprocessed_input)

    # outputs = layers.Concatenate()(preprocessed_inputs)
    outputs = preprocessed_inputs

    mtg_preprocessing = tf.keras.Model(inputs, outputs)

    return mtg_preprocessing, inputs


def build_mtg_model(preprocessing_head, inputs):
    preprocessed_inputs = preprocessing_head(inputs)
    # print(preprocessed_inputs[1])

    x = layers.Concatenate()(preprocessed_inputs)

    x = layers.Dense(units=128, activation="relu")(x)

    outputs = layers.Dense(units=5, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["acc"],
    )
    return model

# Build the preprocessing model
mtg_preprocessing, inputs = build_preprocessing_model(mtg_features)
# Build the training model model = build_mtg_model(mtg_preprocessing, inputs)    

model = build_mtg_model(mtg_preprocessing, inputs)

tf.keras.utils.plot_model(
    model=model,
    show_shapes=True,
    to_file="training_model.png"
)

py
#main.py

import mtg
# Load the data
mtg_features, labels = mtg.load_data()

py
#mtg.py

import pandas as pd


def load_data():
    with open("all_cards.csv", "r", encoding="utf-8") as f:
        df = pd.read_csv(f)

    labels = pd.concat([df.pop(label) for label in ["W", "U", "B", "R", "G"]], axis=1)
    data = df
    
    return (data, labels)

py
def get_output_sequence_length(column):
    """Return the length of the longest sequence of split strings in the column."""
    lengths = [len(x.split()) for x in column]
    return max(lengths)

tf.keras.utils.plot_model(
    model=mtg_preprocessing,
    show_shapes=True,
    to_file="mtg_preprocessin_model.png"
)