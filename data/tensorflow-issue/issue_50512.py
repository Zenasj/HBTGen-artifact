# tf.random.uniform((batch_size, ), dtype=tf.string)  # Assumed input is a dict of string and numeric features keyed by feature names

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Assumptions based on issue:
# - There are categorical features listed in CATEGORICAL_FEATURE_NAMES
# - Each categorical feature has a vocabulary in CATEGORICAL_FEATURES_WITH_VOCABULARY dict
# - Numeric features are also provided and passed as floats
# - The input is a dictionary of tensors keyed by feature names
# - The problem reported is due to incorrect num_oov_indices or input_dim in Embedding layer
#   causing indices to go out of valid range [-1 not allowed, > max index not allowed]

# Example placeholders for missing variables:
CATEGORICAL_FEATURE_NAMES = ['feature_cat1', 'feature_cat2']
# Sample vocabularies for categorical features:
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    'feature_cat1': ['a', 'b', 'c', 'd', 'e'],   # vocab size 5
    'feature_cat2': ['x', 'y', 'z']              # vocab size 3
}
# List of other numeric feature names (as assumed)
NUMERIC_FEATURE_NAMES = ['feature_num1', 'feature_num2']
# Combine all inputs for the model input signature:
ALL_FEATURE_NAMES = CATEGORICAL_FEATURE_NAMES + NUMERIC_FEATURE_NAMES

# Hidden units and dropout rate as an example
hidden_units = [64, 32]
dropout_rate = 0.5

def create_model_inputs():
    # Create input layers for all features (categorical and numeric)
    inputs = {}
    for name in ALL_FEATURE_NAMES:
        if name in CATEGORICAL_FEATURE_NAMES:
            # Categorical features as string inputs:
            inputs[name] = keras.Input(shape=(), name=name, dtype=tf.string)
        else:
            # Numeric features as float32 inputs:
            inputs[name] = keras.Input(shape=(), name=name, dtype=tf.float32)
    return inputs

def encode_inputs(inputs, use_embedding=False):
    """
    Encode inputs dictionary:
    - For categorical features, convert strings with StringLookup:
      - If use_embedding: convert to int indices and embed.
        Use num_oov_indices=1 to allow 1 OOV bucket to avoid -1 indices.
        Embedding input_dim = vocab_size + num_oov_indices to cover all indices.
      - else: output one-hot binary vectors.
    - For numeric features: expand dims and use as-is.
    Finally concatenate along last axis.
    """
    encoded_features = []
    # We set num_oov_indices=1 because the error suggests -1 index which comes from no OOV bucket
    # Mask token is None because user wants no special mask token
    num_oov_indices = 1
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            vocab_size = len(vocabulary)
            lookup = layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=num_oov_indices,
                output_mode="int" if use_embedding else "binary",
            )
            if use_embedding:
                encoded_feature = lookup(inputs[feature_name])  # int indices
                embedding_dims = int(np.ceil(vocab_size ** 0.25))
                # input_dim must include vocab + OOV indices
                embedding = layers.Embedding(
                    input_dim=vocab_size + num_oov_indices,
                    output_dim=embedding_dims,
                    embeddings_initializer="uniform",
                )
                encoded_feature = embedding(encoded_feature)
            else:
                # one-hot binary output_mode returns a vector
                # StringLookup returns shape (batch,), expand dims before one-hot if needed
                encoded_feature = lookup(inputs[feature_name])
            encoded_features.append(encoded_feature)
        else:
            # Numeric feature: float, expand dims to match categorical shape dim
            expanded = tf.expand_dims(inputs[feature_name], -1)
            encoded_features.append(expanded)

    # Concatenate all features on last axis
    # Some features may have shape (batch, dim), some (batch, embedding_dim)
    # This should be compatible if dims align as expected.
    all_features = layers.concatenate(encoded_features)
    return all_features


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create input layers under create_model_inputs (not stored here to keep forward flexible)
        # Hidden layers for deep part
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(layers.Dense(units))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.ReLU())
            self.hidden_layers.append(layers.Dropout(dropout_rate))
        self.output_layer = layers.Dense(units=1, activation="linear")
        self.batchnorm_wide = layers.BatchNormalization()

    def call(self, inputs, training=False):
        # inputs is dict of Tensors keyed by feature names
        # Wide part: categorical features one-hot + numeric as-is
        wide = encode_inputs(inputs, use_embedding=False)
        wide = self.batchnorm_wide(wide, training=training)

        # Deep part: categorical features embedded + numeric as-is
        deep = encode_inputs(inputs, use_embedding=True)
        x = deep
        for layer in self.hidden_layers:
            x = layer(x, training=training)

        merged = layers.concatenate([wide, x])
        outputs = self.output_layer(merged)
        return outputs


def my_model_function():
    # Instantiate and return the Wide & Deep model
    return MyModel()


def GetInput():
    """
    Generate a sample input dictionary matching the expected inputs for MyModel.
    - For categorical features: random selection from their vocabularies as tf.string tensors
    - For numeric features: random float tensors
    Assumptions:
    - batch_size fixed to 4 for sample inputs
    """
    batch_size = 4
    inputs = {}
    for feature_name in ALL_FEATURE_NAMES:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            vocab = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Randomly choose vocab strings as batch input
            random_indices = np.random.randint(0, len(vocab), size=batch_size)
            random_strings = np.array(vocab)[random_indices]
            inputs[feature_name] = tf.convert_to_tensor(random_strings, dtype=tf.string)
        else:
            # Numeric features: random floats
            inputs[feature_name] = tf.random.uniform(shape=(batch_size,), minval=0, maxval=1, dtype=tf.float32)
    return inputs

