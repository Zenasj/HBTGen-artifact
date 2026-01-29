# tf.random.uniform((B, max_len), dtype=tf.float32) ← Input is batch of sentence word indices (as floats due to original code dtype='float32')

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, keras

embedding_dim = 100  # assumed embedding dimension, can be changed as required

class MyModel(tf.keras.Model):
    def __init__(self, word_to_vec_map, word_to_index, max_len):
        super().__init__()
        self.vocab_len = len(word_to_index) + 1  # +1 for padding or unknown token

        # Prepare embedding matrix from word vectors
        emb_matrix = np.zeros((self.vocab_len, embedding_dim))
        for word, idx in word_to_index.items():
            emb_matrix[idx, :] = word_to_vec_map[word]
        
        # Embedding layer: pretrained, non-trainable
        self.embedding_layer = layers.Embedding(
            input_dim=self.vocab_len,
            output_dim=embedding_dim,
            trainable=False,
            mask_zero=False  # mask_zero False since zero is used for unknown words
        )

        # Build embedding layer and set weights
        self.embedding_layer.build((None,))  # required before set_weights
        self.embedding_layer.set_weights([emb_matrix])

        # LSTM and prediction layers
        self.lstm = layers.LSTM(128)
        self.dropout = layers.Dropout(0.5)
        # 2 output units with sigmoid activation (could be binary classification)
        self.dense = layers.Dense(2, activation='sigmoid', name="predictions")

    def call(self, inputs, training=False):
        # inputs expected shape: (batch_size, max_len)
        # originally dtype=float32 because Input layer in original code used float32
        # but embedding lookup expect integer indices, so cast to int32 and replace unknowns with 0
        
        # Defensive: cast inputs to int32
        indices = tf.cast(inputs, tf.int32)

        # Clip negative or invalid indices to zero (unknown token)
        # This is to avoid InvalidArgumentError on embeddings
        indices = tf.where(indices < 0, tf.zeros_like(indices), indices)

        x = self.embedding_layer(indices)
        x = self.lstm(x)
        if training:
            x = self.dropout(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # As dictionary examples are required to build the model, 
    # use dummy minimal data as placeholders with realistic dimensions for demonstration.
    # In real use, user would supply actual word_to_vec_map and word_to_index

    # Minimal example map: word_to_index with one word, 'hello'
    word_to_index = {'hello': 1}
    max_len = 10  # example maximum sentence length

    # word_to_vec_map: map words to random vectors of shape (embedding_dim,)
    word_to_vec_map = {'hello': np.random.rand(embedding_dim)}

    model = MyModel(word_to_vec_map, word_to_index, max_len)
    # Build the model by calling on a sample input
    sample_input = tf.zeros((1, max_len), dtype=tf.float32)
    _ = model(sample_input, training=False)
    return model

def GetInput():
    # Generate random input tensor of shape (batch_size, max_len)
    # Using batch size = 32, max_len = 10 (matches my_model_function defaults)
    batch_size = 32
    max_len = 10
    # Indices range 0 to vocab_len-1, but unknown words set to 0
    # Here, simulate valid indices with some zeros for unknowns
    # Random integers between 0 and 2 (0 unknown, 1 a known word)
    random_indices = np.random.randint(0, 2, size=(batch_size, max_len))
    # Cast to float32 because the model Input expects float32 dtype (per original code)
    return tf.convert_to_tensor(random_indices, dtype=tf.float32)

