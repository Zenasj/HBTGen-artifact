import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf


def test_tensorflow_concatenate(inputs):
    tf.concat(inputs, axis=-1)

    print("tf.concat works with {} inputs".format(len(inputs)))


def test_concatenate_layer_with_inputs(inputs):
    model = tf.keras.Sequential((
        tf.keras.layers.Concatenate(axis=-1),
        tf.keras.layers.Dense(32)))

    feed_dict = {
        input_: np.random.uniform(
            0, 1, (3, *input_.shape[1:].as_list()))
        for input_ in inputs
    }
    output = model(inputs)
    output_eval = tf.keras.backend.get_session().run(
        output, feed_dict=feed_dict)
    output_np = model.predict([feed_dict[key] for key in inputs])

    assert np.allclose(output_eval, output_np)

    print("tf.keras.layers.Concatenate with {} inputs".format(len(inputs)))


def main():
    input1 = tf.keras.layers.Input((1, ))
    input2 = tf.keras.layers.Input((2, ))

    test_tensorflow_concatenate([input1, input2])
    test_tensorflow_concatenate([input1])

    test_concatenate_layer_with_inputs([input1, input2])
    test_concatenate_layer_with_inputs([input1])


if __name__ == '__main__':
    main()