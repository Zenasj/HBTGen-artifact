import random
from tensorflow import keras
from tensorflow.keras import layers

# list available GPUS, make sure you have at least two
gpus = tf.config.experimental.list_logical_devices('GPU')
assert len(gpus) >= 2
# place the embedding layer on the first GPU
with tf.device(gpus[0]):
    embedded = tf.keras.layers.Embedding(input_dim, output_dim)(inputs)
# place the rest of the model on the second GPU
with tf.device(gpus[1]):
    output = some_layers_stack(embedded)

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=366856, output_dim=1000, input_length=4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1000)),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(64, activation='softmax')
])
model.compile('adam', 'sparse_categorical_crossentropy', ['sparse_categorical_accuracy'])

mock_inputs = tf.random.uniform((64, 4), 1, 366856, tf.int64)
mock_target = tf.random.uniform((64, 1), 0, 64, tf.int64)

model.fit(mock_inputs, mock_target, batch_size=32, epochs=5)

class SharedKernelSoftmax(tf.keras.layers.Layer):
    def __init__(self, kernel, bias_initializer='zeros'):
        self.kernel = kernel
        self.bias = self.add_weight(
            name='bias', shape=(tf.shape(kernel)[1],), dtype=kernel.dtype,
            initializer=bias_initializer
        )

    def call(self, inputs, **kwargs):
        output = tf.keras.backend.dot(inputs, self.kernel) + self.bias
        return tf.nn.softmax(output, axis=-1)