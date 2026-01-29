# tf.random.uniform((B, None), dtype=tf.int32) ← Input is a batch of variable-length sequences of integers

import tensorflow as tf
from tensorflow.keras import backend as K

class CustomEmbedding(tf.keras.layers.Layer):
    def __init__(self, masking_boolean, vocab, dimension, **kwargs):
        # masking_boolean is expected to be a boolean tensor mask (shape compatible with input ids)
        self.masking_boolean = masking_boolean
        self.vocab = vocab
        self.dimension = dimension
        super(CustomEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create embedding weights for vocab x dimension
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.vocab, self.dimension),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True)
        super(CustomEmbedding, self).build(input_shape)

    def call(self, inputs):
        # Use tf.nn.embedding_lookup to get embeddings for input ids
        return tf.nn.embedding_lookup(params=self.kernel, ids=inputs)

    def compute_mask(self, inputs, mask=None):
        # Return the mask provided at layer instantiation
        return self.masking_boolean

    def compute_output_shape(self, input_shape):
        # Output shape: (batch_size, seq_len, dimension)
        # K.shape returns dynamic shapes (tensors), so this returns a shape tuple with tensors
        return (K.shape(input_shape)[0], K.shape(input_shape)[1], self.dimension)


class MyModel(tf.keras.Model):
    def __init__(self, vocab=500, dimension=512, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # We will create the embedding layer inside the call method because the mask depends on input.
        # But to avoid recreating variables repeatedly, create an embedding weight variable here.
        self.vocab = vocab
        self.dimension = dimension
        # Create embedding kernel weight once:
        self.embeddings = self.add_weight(
            name='embedding_kernel',
            shape=(self.vocab, self.dimension),
            initializer='glorot_uniform',
            trainable=True)

    def call(self, inputs):
        # inputs: int32 tensor of shape (batch_size, seq_len) with word indices.
        # Create mask for non-zero entries in the input (boolean tensor)
        mask_boolean = tf.not_equal(inputs, 0)  # shape (batch_size, seq_len), bool
        mask_float = tf.cast(mask_boolean, tf.float32)  # float mask for multiplication

        # Use embedding lookup with the shared embedding weights
        embeddings = tf.nn.embedding_lookup(params=self.embeddings, ids=inputs)  # (B, seq_len, dim)

        # Apply mask (zero-out embeddings for padded positions)
        masked_embeddings = embeddings * tf.expand_dims(mask_float, axis=-1)

        # For compliance with original compute_mask semantics, store mask as attribute
        self.current_mask = mask_boolean

        return masked_embeddings

    def compute_mask(self, inputs, mask=None):
        # Return the current mask computed in call
        # Because Keras calls compute_mask after call, self.current_mask should be set
        return getattr(self, 'current_mask', None)


def my_model_function():
    # Return a new instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random batch of integer sequences:
    # batch size: 4, sequence length: variable but choose 7 here for a consistent shape
    # values in [0, vocab-1], where 0 is used as padding token
    batch_size = 4
    seq_len = 7
    vocab = 500

    # Generate random integers with some zeros to simulate padding:
    import numpy as np
    np.random.seed(42)
    data = np.random.randint(1, vocab, size=(batch_size, seq_len))
    # introduce zeros as padding in last positions of some sequences
    data[0, 5:] = 0
    data[2, 3:] = 0

    # Convert to tf.Tensor
    tensor_input = tf.convert_to_tensor(data, dtype=tf.int32)
    return tensor_input

