import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf


def build_autoencoder(input_dim, embed_dim=100): 
    """Set up an auto-encoder model made of two BiLSTM layers.""" 
    # Set up input tensors.
    inputs = tf.keras.Input((None, input_dim), dtype=tf.float32) 
    mask = tf.keras.Input((None,), dtype=tf.bool) 
    # Set up encoder and decoder BiLSTM layers.
    encoder = tf.keras.layers.Bidirectional( 
        tf.keras.layers.LSTM(embed_dim, return_sequences=True),
        merge_mode='sum' 
    ) 
    decoder = tf.keras.layers.Bidirectional( 
        tf.keras.layers.LSTM(input_dim, return_sequences=True),
        merge_mode='sum' 
    ) 
    # Build the outputs tensor.
    outputs = decoder(encoder(inputs, mask=mask), mask=mask) 
    # Set up, compile and return the model.
    model = tf.keras.Model(inputs=[inputs, mask], outputs=outputs) 
    model.compile('adam', tf.keras.losses.mse) 
    return model


def build_mock_data(dim, nsamples, maxlen, seed=0):
    """Build some mock data for bug demonstration purposes.

    Return an array of zero-padded sequences of random
    actual length, and an associated boolean mask Tensor.
    
    Use a random seed for reproducibility.
    """
    np.random.seed(seed)
    sizes = np.random.choice(maxlen, size=nsamples)
    inputs = np.random.normal(size=(nsamples, max(sizes), dim))
    for i, size in enumerate(sizes):
        inputs[i, size:] = 0.
    mask = tf.sequence_mask(sizes, dtype=tf.bool)
    return inputs.astype(np.float32), mask


if __name__ == '__main__':
    # Generate the mock data. Instantiate the mdoel.
    inputs, mask = build_mock_data(dim=100, nsamples=64, maxlen=500, seed=0)
    model = build_autoencoder(input_dim=100, embed_dim=50)

    # This works fine.
    model.predict([inputs, mask])

    # This also works.
    model.evaluate([inputs, mask], inputs)

    # This is where things go wrong.
    model.fit([inputs, mask], inputs)

import numpy as np
import tensorflow as tf


def build_autoencoder(input_dim, embed_dim=100): 
    """Set up an auto-encoder model made of two BiLSTM layers.""" 
    # Set up the input tensor.
    inputs = tf.keras.Input((None, input_dim), dtype=tf.float32) 
    # Set up encoder and decoder BiLSTM layers.
    encoder = tf.keras.layers.Bidirectional( 
        tf.keras.layers.LSTM(embed_dim, return_sequences=True),
        merge_mode='sum', name='encoder'
    ) 
    decoder = tf.keras.layers.Bidirectional( 
        tf.keras.layers.LSTM(input_dim, return_sequences=True),
        merge_mode='sum', name='decoder'
    ) 
    # Build the outputs tensor.
    outputs = decoder(encoder(inputs)) 
    # Set up, compile and return the model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs) 
    model.compile('adam', tf.keras.losses.mse) 
    return model


def build_mock_data(dim, nsamples, length, seed=0):
    """Build some mock data for bug demonstration purposes.

    Return an array of shape (nsamples, length, dim) filled
    with random normally-distributed data.
    
    Use a random seed for reproducibility.
    """
    np.random.seed(seed)
    return np.random.normal(size=(nsamples, length, dim))


if __name__ == '__main__':
    # Generate the mock data. Instantiate the mdoel.
    inputs = build_mock_data(dim=100, nsamples=64, length=500, seed=0)
    model = build_autoencoder(input_dim=100, embed_dim=50)

    # This works but prints the error warning.
    model.predict(inputs)

    # Same thing here.
    model.evaluate(inputs, inputs)

    # Same thing here.
    model.fit(inputs, inputs)

def build_autoencoder(input_dim, embed_dim=100): 
    """Set up an auto-encoder model made of two BiLSTM layers.""" 
    # Set up the input tensors.
    inputs = tf.keras.Input((None, input_dim), dtype=tf.float32)
    sizes = tf.keras.Input((), dtype=tf.int32)
    # Set up encoder and decoder BiLSTM layers.
    encoder = tf.keras.layers.Bidirectional( 
        tf.keras.layers.LSTM(embed_dim, return_sequences=True),
        merge_mode='sum', name='encoder'
    ) 
    decoder = tf.keras.layers.Bidirectional( 
        tf.keras.layers.LSTM(input_dim, return_sequences=True),
        merge_mode='sum', name='decoder'
    ) 
    # Build the outputs tensor.
    mask = tf.sequence_mask(sizes, maxlen=tf.shape(inputs)[1])
    outputs = decoder(encoder(inputs, mask=mask), mask=mask) 
    # Set up, compile and return the model.
    model = tf.keras.Model(inputs=[inputs, sizes], outputs=outputs) 
    model.compile('adam', tf.keras.losses.mse) 
    return model

def build_mock_data(dim, nsamples, maxlen, seed=0):
    """Build some mock data for bug demonstration purposes.

    Return an array of zero-padded sequences of random
    actual length, and an array containing those lengths.
    
    Use a random seed for reproducibility.
    """
    np.random.seed(seed)
    sizes = np.random.choice(maxlen, size=nsamples)
    inputs = np.random.normal(size=(nsamples, max(sizes), dim))
    for i, size in enumerate(sizes):
        inputs[i, size:] = 0.
    return inputs.astype(np.float32), sizes

if __name__ == '__main__':
    # Generate the mock data. Instantiate the mdoel.
    inputs, sizes = build_mock_data(dim=100, nsamples=64, maxlen=500, seed=0)
    model = build_autoencoder(input_dim=100, embed_dim=50)

    # This works fine.
    model.predict([inputs, sizes])

    # This also works.
    model.evaluate([inputs, sizes], inputs)

    # This prints out the error messages, but works.
    model.fit([inputs, sizes], inputs)

    # Further calls no longer print errors, and the loss decreases.
    model.fit([inputs, sizes], inputs)
    model.fit([inputs, sizes], inputs)
    model.fit([inputs, sizes], inputs)