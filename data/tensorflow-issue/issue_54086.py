import tensorflow as tf
from tensorflow import keras

from typing import Tuple

import tensorflow.compat.v1 as tf
from tensorflow.keras import models
from tensorflow.keras import layers

tf.disable_v2_behavior()


def create_model() -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Create a model with Neural-Net: 1024 x 1024
    """
    graph = tf.get_default_graph()
    with graph.as_default():
        # The Neural Net Size: 200 -> 1024 x 1024 -> 6
        model = models.Sequential()
        input_shape, hidden_shape, output_shape = 200, 1024, 6
        model.add(
            layers.Dense(
                hidden_shape,
                activation='tanh',
                input_shape=(input_shape,),
                name="layer0"
            )
        )
        model.add(layers.Dense(hidden_shape, activation='tanh', name="layer1"))
        model.add(layers.Dense(output_shape, activation='relu', name="final_layer"))
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, input_shape])
        output_tensor = model(input_tensor)
    return input_tensor, output_tensor


def save_model(sess: tf.Session, path: str) -> None:
    """
    Save the TF model according to the path.
    """
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    save_path = saver.save(sess, path)
    print(f"Saved the model in {save_path}")


def main() -> None:
    input_tensor, output_tensor = create_model()
    print(
        f"Create the model with Input tensor {input_tensor} "
        f"and output tensor {output_tensor}."
    )
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    save_model(sess, "./test_model_repro/foo")


main()

import os
import psutil

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def _get_memory_rss() -> float:
    """
    Get the RSS memory value in GB.
    """
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3


def restore(path: str) -> tf.train.Saver:
    """
    Restores the TF graph from the path and returns the saver.
    """
    sess = tf.Session()
    saver = tf.train.import_meta_graph(f"{path}.meta")
    saver.restore(sess, path)


def main() -> None:
    init_prev = _get_memory_rss()
    for _ in range(10000):
        restore("./test_model_repro/foo")
        mem_after = _get_memory_rss() - init_prev
        print(
            f"The memory increased after restoring the TF model: {mem_after}"
        )
        tf.reset_default_graph()
        tf.keras.backend.clear_session()


main()