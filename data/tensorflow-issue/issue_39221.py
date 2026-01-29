# tf.random.uniform((B,)) with dictionary keys 'f1' and 'f2' as string categorical features

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import feature_column

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define categorical feature columns as in provided example
        # f1: categorical_column_with_vocabulary_list with vocab ['x1','x2','x3']
        categorical_f1 = feature_column.categorical_column_with_vocabulary_list(
            key='f1', vocabulary_list=['x1','x2','x3'], num_oov_buckets=0)
        one_hot_f1 = feature_column.indicator_column(categorical_f1)

        # f2: categorical_column_with_vocabulary_list with vocab ['x1','x2','x3','x4','x5','x6']
        # with 5 OOV buckets and embedding dimension 10
        categorical_f2 = feature_column.categorical_column_with_vocabulary_list(
            key='f2', vocabulary_list=['x1','x2','x3','x4','x5','x6'], num_oov_buckets=5)
        embedding_f2 = feature_column.embedding_column(categorical_f2, dimension=10)

        self.feature_columns = [one_hot_f1, embedding_f2]
        self.feature_layer = tf.keras.layers.DenseFeatures(self.feature_columns)

        # Dense layers as per example
        self.dense1 = layers.Dense(20, activation='relu')
        self.dense2 = layers.Dense(1, activation='softsign')

    def call(self, inputs, training=False):
        # inputs: dictionary of tensors with keys 'f1' and 'f2', each tensor shape (B,)
        x = self.feature_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model with same optimizer, loss and metrics as in example
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a dictionary of inputs matching the feature columns expected by MyModel.
    # Each feature is a batch of string tensors.
    # For simplicity, batch size = 8.
    batch_size = 8

    # f1 categories ['x1', 'x2', 'x3']
    # Generate random choices from list as string tensors
    f1_choices = tf.constant(['x1', 'x2', 'x3'])
    f1_idx = tf.random.uniform(shape=(batch_size,), minval=0, maxval=3, dtype=tf.int32)
    f1_input = tf.gather(f1_choices, f1_idx)

    # f2 categories ['x1','x2','x3','x4','x5','x6'] plus OOV buckets (handled internally)
    f2_choices = tf.constant(['x1','x2','x3','x4','x5','x6'])
    f2_idx = tf.random.uniform(shape=(batch_size,), minval=0, maxval=6, dtype=tf.int32)
    f2_input = tf.gather(f2_choices, f2_idx)

    inputs = {
        'f1': f1_input,
        'f2': f2_input
    }
    return inputs

