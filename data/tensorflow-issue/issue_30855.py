# tf.random.uniform((B, SEQ_LEN, FEATURES), dtype=tf.float32) ← assuming input shape typical for sequence models (e.g. LSTM) from the discussion about Bidirectional LSTM and padded batching

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This fused model encapsulates two versions of a model:
        # 1) A subclassed Keras model with some custom layers (e.g. dynamic custom pooling layer)
        # 2) A Functional API model (replicated here for comparison)
        #
        # Given multiple comments on differences in execution mode and parallelism,
        # this fused model returns comparison info between the two.

        # For simplification and to infer the models:
        # Subclassed model (mimicking Bidirectional LSTM + custom pooling layer)
        self.sub_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, name='sub_lstm_inner'),
            name='sub_bi_lstm')
        # Mimic custom pooling with a dynamic approach (here just GlobalAveragePooling1D for placeholder)
        self.sub_pool = tf.keras.layers.GlobalAveragePooling1D(name='sub_pool')

        # Functional-like model: same layers, using ReLU activations to slightly differentiate
        self.func_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, name='func_lstm_inner'),
            name='func_bi_lstm')
        self.func_pool = tf.keras.layers.GlobalMaxPooling1D(name='func_pool')

        # Output layers for both branches
        self.sub_dense = tf.keras.layers.Dense(64, activation='relu', name='sub_dense')
        self.func_dense = tf.keras.layers.Dense(64, activation='relu', name='func_dense')

        # Final output layer shared (could be for classification/regression)
        self.out = tf.keras.layers.Dense(10, activation='softmax', name='output')

    def call(self, inputs, training=False):
        # Forward through subclassed model branch
        x_sub = self.sub_lstm(inputs)
        # Assume dynamic operations fixed by using tf.shape in a real custom layer replaced here with GAP1D
        x_sub = self.sub_pool(x_sub)
        x_sub = self.sub_dense(x_sub)

        # Forward through functional model branch
        x_func = self.func_lstm(inputs)
        x_func = self.func_pool(x_func)
        x_func = self.func_dense(x_func)

        # Compare outputs numerically (e.g., L2 norm of difference)
        diff = tf.norm(x_sub - x_func, ord='euclidean', axis=-1)  # shape (batch,)
        # Boolean tensor where they are close (within tolerance)
        close = tf.less(diff, 1e-3)

        # Combine final outputs (pass through shared output layer)
        out_sub = self.out(x_sub)
        out_func = self.out(x_func)

        # Return a dict with all relevant info for inspection
        return {
            'out_sub': out_sub,
            'out_func': out_func,
            'diff_norm': diff,
            'close': close
        }

def my_model_function():
    # Return instance of fused MyModel
    return MyModel()

def GetInput():
    # Based on typical input for Bidirectional LSTM with padded batching
    # Batch size: 8
    # Sequence length: 100 (variable-length sequence padded to fixed length)
    # Features: 32 (e.g., embedding dimension)
    B, SEQ_LEN, FEATURES = 8, 100, 32
    x = tf.random.uniform((B, SEQ_LEN, FEATURES), dtype=tf.float32)
    return x

