from tensorflow.keras import models
from tensorflow.keras import optimizers

import random
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_datasets.core.features.text import SubwordTextEncoder

EOS = '<eos>'
PAD = '<pad>'

RESERVED_TOKENS = [EOS, PAD]
EOS_ID = RESERVED_TOKENS.index(EOS)
PAD_ID = RESERVED_TOKENS.index(PAD)

dictionary = [
    'verstehen',
    'verstanden',
    'vergessen',
    'verlegen',
    'verlernen',
    'vertun',
    'vertan',
    'verloren',
    'verlieren',
    'verlassen',
    'verhandeln',
]

dictionary = [word.lower() for word in dictionary]


def get_model(params) -> keras.models.Model:

    inputs = layers.Input((None,), dtype=tf.int64, name='inputs')

    x = inputs

    vocab_size = params['vocab_size']
    hidden_size = params['hidden_size']
    max_input_length = params['max_input_length']
    max_target_length = params['max_target_length']

    x = layers.Embedding(vocab_size, hidden_size, input_length=max_input_length)(x)

    # Encoder
    x = layers.RNN(layers.GRUCell(hidden_size))(x)
    x = layers.RepeatVector(max_target_length)(x)

    # Deoder
    x = layers.RNN(layers.GRUCell(hidden_size), return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(hidden_size, activation='relu'))(x)

    # Outputs
    output_dense_layer = layers.Dense(vocab_size, activation='softmax')
    outputs = layers.TimeDistributed(output_dense_layer, name='outputs')(x)

    return keras.models.Model(inputs=[inputs], outputs=[outputs])


def sample_generator(text_encoder: SubwordTextEncoder, max_sample: int = None):
    count = 0

    while True:
        random.shuffle(dictionary)

        for word in dictionary:

            for i in range(1, len(word)):

                inputs = word[:i]
                targets = word

                example = dict(
                    inputs=text_encoder.encode(inputs) + [EOS_ID],
                    targets=text_encoder.encode(targets) + [EOS_ID],
                )
                count += 1

                yield example

                if max_sample is not None and count >= max_sample:
                    print('Reached max_samples (%d)' % max_sample)
                    return


def make_dataset(generator_fn, params, training):

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_types={
            'inputs': tf.int64,
            'targets': tf.int64,
        }
    )

    if training:
        dataset = dataset.shuffle(100)

    dataset = dataset.padded_batch(
        params['batch_size'],
        padded_shapes={
            'inputs': (None,),
            'targets': (None,)
        },
    )

    if training:
        dataset = dataset.map(lambda example: to_train_example(example, params=params)).repeat()

    return dataset


def to_train_example(example: dict, params: dict):
    # Make sure targets are one-hot encoded
    example['targets'] = tf.one_hot(example['targets'], depth=params['vocab_size'])
    return example


def main():

    text_encoder = SubwordTextEncoder.build_from_corpus(
        iter(dictionary),
        target_vocab_size=1000,
        max_subword_length=6,
        reserved_tokens=RESERVED_TOKENS
    )

    generator_fn = partial(sample_generator, text_encoder=text_encoder, max_sample=10)

    params = dict(
        batch_size=20,
        vocab_size=text_encoder.vocab_size,
        hidden_size=32,
        max_input_length=30,
        max_target_length=30,
        enable_metrics_in_training=True
    )

    model = get_model(params)

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
    )

    assert len(model.trainable_variables), 'There are no trainable_variables'
    model.summary()

    train_dataset = make_dataset(generator_fn, params, training=True)

    model.fit(
        train_dataset,
        epochs=5,
        steps_per_epoch=100,
    )


if __name__ == '__main__':
    main()

import tensorflow as tf
from tensorflow import keras


def xor_data():
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]
    while True:
        for x, y in zip(inputs, targets):
            yield x, y


class FeedForwardNetwork(keras.models.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layers = [
            keras.layers.Dense(4, activation='sigmoid'),
            keras.layers.Dense(4, activation='sigmoid'),
            keras.layers.Dense(1, activation='sigmoid')
        ]

    def call(self, x, **kwargs):
        for layer in self._layers:
            x = layer(x)
        return x


class XorCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        all_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = self.model.predict(all_data)

        print('\nPredictions: ')
        print((y > 0.5) * 1)


def main():

    dataset = tf.data.Dataset.from_generator(xor_data, output_types=(tf.int64, tf.int64)).batch(10).shuffle(100)

    train_dataset = dataset.repeat()
    dev_dataset = dataset

    for batch in dataset:
        print(batch)
        break

    model_internal = FeedForwardNetwork()

    inputs = keras.layers.Input(shape=(2,))
    logits = model_internal(inputs)

    model = keras.models.Model(inputs=[inputs], outputs=[logits])

    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss='mse',
        metrics=['mse']
    )

    model.summary()

    model_fp = '/tmp/xor/model'

    callbacks = [
        keras.callbacks.ModelCheckpoint(
          model_fp,
          save_best_only=True,
          save_weights_only=False
        ),
        XorCallback()
    ]

    model.fit(
        train_dataset,
        epochs=5,
        steps_per_epoch=5000,
        validation_data=dev_dataset,
        validation_steps=100,
        callbacks=callbacks
    )


if __name__ == '__main__':
    main()