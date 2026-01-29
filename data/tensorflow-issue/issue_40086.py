# tf.random.uniform((4,), dtype=tf.string) ← Input is a batch of 4 string tokens for categorical vocabulary lookup

import numpy as np
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        vocabulary = ['word1', 'word2']
        # Create categorical column with fixed vocabulary
        categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
            'feat1', vocabulary)
        # Embedding column with embedding dimension 1, initialized to zeros
        embedding_column = tf.feature_column.embedding_column(
            categorical_column,
            dimension=1,
            initializer=tf.constant_initializer(0))
        # DenseFeatures layer to handle feature columns
        self.feature_layer = tf.keras.layers.DenseFeatures([embedding_column])
        # Final Dense layer, units=1, no bias, not trainable, kernel initialized to 1 (constant)
        self.dense_output = tf.keras.layers.Dense(
            units=1,
            use_bias=False,
            trainable=False,
            kernel_initializer=tf.constant_initializer(1))

    def call(self, inputs):
        """
        inputs: dict with key 'feat1' mapping to a tensor of strings shape (batch,)
        Outputs logits of shape (batch, 1)
        """
        x = self.feature_layer(inputs)
        x = self.dense_output(x)
        return x

def my_model_function():
    # Instantiate the model and run once to initialize variables
    model = MyModel()
    # Provide dummy input to ensure variables get created before set_weights or training
    dummy_input = GetInput()
    model(dummy_input)
    return model

def GetInput():
    # Return a dictionary with key 'feat1' mapping to a batch of 4 strings
    # matching the vocabulary ['word1', 'word2'] used in MyModel
    # This matches the example input in the issue.
    batch = np.array(['word1', 'word1', 'word2', 'word2'])
    return {'feat1': batch}

