from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,)),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='sgd', loss='mse')
    return model


def do_test():
    x = np.asarray([[0], [1]])
    y = x
    model = build_model()
    model.fit(x=x, y=y, epochs=1000, verbose=0)
    model.get_weights()
    result = model.predict(x=x)
    print('result:', result)
    print('result close?:', np.allclose(result, y, atol=0.01))


def test_with_graph_static_execution():
    with tf.Graph().as_default():  # pylint: disable=not-context-manager
        do_test()


def test_without_explicit_graph_using_static_execution():
    disable_eager_execution()
    do_test()


def test_without_graph_eager_execution():
    do_test()


test_with_graph_static_execution()
# test_without_explicit_graph_using_static_execution()
# test_without_graph_eager_execution()

def test_without_explicit_graph_using_disable_eager_execution():
    disable_eager_execution()
    do_test()

def test_without_explicit_graph_using_tf_function():
    tf.function(do_test())