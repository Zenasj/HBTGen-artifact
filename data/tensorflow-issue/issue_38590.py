# tf.random.uniform((B, 1), dtype=tf.string)  ← Input shape inference: batch size B, single string per example

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np

# Character dictionary used for vocabulary; adjusted to avoid escaping issues and length mismatch
CHAR_DICT = 'abcdefghijklmnopqrstuvwxyz0123456789 .!?:,\'%-()/$|&;[]"'

embedding_dim = 128  # Embedding dimension constant used in vectorize submodel


@tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
@tf.function
def split_text(text_list):
    # Custom function to split strings into characters expected by TextVectorization
    # text_list: a scalar or vector of strings; returns shape (1, seq_len)
    joined = tf.strings.reduce_join(text_list)
    split = tf.strings.unicode_split(joined, 'UTF-8')
    return tf.expand_dims(split, 0)  # Add batch dim as required


def vectorize_layers(factor):
    # Builds a submodel that vectorizes text input and applies embedding, dense and dropout layers
    text_input = layers.Input(shape=(1,), dtype=tf.string)
    vectorize_layer = TextVectorization(
        split=split_text,
        max_tokens=len(CHAR_DICT) + 1,  # +1 for OOV or masking token
        output_mode='int',
        output_sequence_length=128
    )
    vectorize_layer.set_vocabulary(list(CHAR_DICT))
    x = vectorize_layer(text_input)
    x = layers.Embedding(len(CHAR_DICT) + 1, embedding_dim)(x)
    x = layers.Dense(factor, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    # Note: Avoid flatten here because inside TimeDistributed that causes output mismatch
    model = tf.keras.Model(text_input, x)
    return model


class MyModel(tf.keras.Model):
    def __init__(self, input_labels=["a", "b"], factor=10):
        super().__init__()
        self.input_labels = input_labels
        self.factor = factor
        # Build vectorize submodel once
        self.vectorize_submodel = vectorize_layers(factor)
        # Prepare to accept multiple named string inputs
        self.flatten = layers.Flatten()
        self.concat = layers.Concatenate()
        self.dense1 = layers.Dense(factor)
        self.dense2 = layers.Dense(1, activation='sigmoid', name='predictions')

    def call(self, inputs):
        # inputs is a dict of string tensor with shape (batch, 1)
        # Extract inputs for each label, shape=(batch, 1)
        subs = []
        for label in self.input_labels:
            # Expect inputs[label] shape (batch, 1), tf.string dtype
            inp = inputs[label]
            # Expand dims for channel if needed for TimeDistributed, here we add axis=-1
            # We need shape (batch, 1, 1), so we expand dims on axis=-1
            x = tf.expand_dims(inp, -1)
            # Send through TimeDistributed vectorize model
            # Actually vectorize_submodel expects (batch, 1), so remove TimeDistributed and assume single time step
            # But original user code tried TimeDistributed over multiple inputs concatenated - we replicate that logic here
            # So to fuse multiple inputs:
            # -> concatenate inputs along the last dimension (axis=-1) after expanding dims
            subs.append(inp)

        # Concatenate input strings along last dimension axis
        concat_inputs = self.concat(subs)  # shape (batch, num_inputs)
        # Expand last dim for apply vectorize submodel via TimeDistributed
        # But vectorize_layers expects (batch, 1) shape, so instead we split along the last dim and map vectorize_submodel
        # Simulate TimeDistributed by mapping over last dim (axis=1)
        # Instead of complex logic, implement a simple loop tf.vectorized_map is tricky for strings
        # We'll stack and map manually via tf.map_fn
        # But given complexity, do per input separately and stack after embedding

        # Alternatively, as original code used layers.TimeDistributed(vectorize_layers)(x),
        # where x was concatenated strings expanded by axis -1
        # So we recreate that:
        # x = tf.expand_dims(concat_inputs, -1)  # shape (batch, time, 1)
        # Then apply TimeDistributed
        x = tf.expand_dims(concat_inputs, -1)  # shape (batch, time, 1)
        x = tf.keras.layers.TimeDistributed(self.vectorize_submodel)(x)  # output shape (batch, time, seq_len, embedding_dim) with factor dense dims

        # Flatten to prepare for final dense layers
        x = self.flatten(x)  # shape (batch, ...)

        x = self.dense1(x)
        x = self.dense2(x)
        return x


def my_model_function():
    # Instantiate MyModel with default input labels and factor
    return MyModel(input_labels=["a", "b"], factor=10)


def GetInput():
    # Construct sample input dictionary with inputs for all required labels
    batch_size = 2  # Reasonable batch size for demonstration

    # Each input is a tensor of shape (batch, 1) dtype string
    # Provide example strings compatible with CHAR_DICT and the vectorizer preprocessor
    a_input = tf.constant([
        "hello world!", 
        "test input"
    ], shape=(batch_size, 1), dtype=tf.string)

    b_input = tf.constant([
        "tensorflow is fun", 
        "keras layers"
    ], shape=(batch_size, 1), dtype=tf.string)

    # Return dict keyed by input label names matching MyModel's expectation
    return {"a": a_input, "b": b_input}

