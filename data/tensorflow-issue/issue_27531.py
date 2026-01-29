# tf.random.uniform((B, T, Features), dtype=tf.float32) ← Assumed input shape: batch size B, sequence length T, feature dimensions

import tensorflow as tf
from tensorflow.keras import layers

# The issue discusses SequenceFeatures layer missing the `_is_feature_layer` property,
# which causes errors when used in keras.Sequential models expecting feature layers.
# This snippet reconstructs a minimal compatible MyModel that uses a SequenceFeatures-like layer
# subclass with the `_is_feature_layer` property fixed, and a simple LSTM + Dense pipeline.

# We create a dummy SequenceFeatures layer subclass with `_is_feature_layer` properly set,
# mimicking the behavior described, but minimal and compatible for demonstration.

class FixedSequenceFeaturesLayer(layers.Layer):
    def __init__(self, output_dim=8, **kwargs):
        # output_dim is arbitrary, simulating total_elements after feature columns processing
        super().__init__(**kwargs)
        self.output_dim = output_dim
    
    @property
    def _is_feature_layer(self):
        # This fixes the issue mentioned in the original Github issue
        return True

    def build(self, input_shape):
        # No weights for this dummy feature extractor
        super().build(input_shape)

    def call(self, inputs):
        # Inputs shape assumed (batch_size, sequence_length, feature_dim)
        # Simulate feature transformation returning shape (batch_size, sequence_length, output_dim)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        # Just a linear projection for demo (simulate transformation)
        x = tf.keras.layers.Dense(self.output_dim)(inputs)
        return x

    def compute_output_shape(self, input_shape):
        # Output shape: (batch_size, sequence_length, output_dim)
        return (input_shape[0], input_shape[1], self.output_dim)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use the fixed SequenceFeatures layer with _is_feature_layer property
        self.feature_layer = FixedSequenceFeaturesLayer(output_dim=8, name="fixed_sequence_features")
        # Two LSTM layers per original example, note corrected arg return_sequences for first
        self.lstm1 = layers.LSTM(1, return_sequences=True, name="lstm1")
        self.lstm2 = layers.LSTM(1, name="lstm2")
        # Dense output layer with relu activation
        self.dense = layers.Dense(1, activation='relu', name='output')

    def call(self, inputs, training=False):
        x = self.feature_layer(inputs)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor simulating sequence data
    # Assumptions:
    # batch size (B) = 4
    # sequence length (T) = 10
    # per-step feature dimension = 5
    B = 4
    T = 10
    feature_dim = 5
    # Using float32 dtype
    return tf.random.uniform((B, T, feature_dim), dtype=tf.float32)

