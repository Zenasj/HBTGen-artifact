import random
import tensorflow as tf

import argparse
from typing import Tuple

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import models
from tensorflow.keras import layers

tf.disable_v2_behavior()


def build_model(input_shape: int, output_shape: int) -> Tuple[tf.Tensor, tf.Tensor]:
    graph = tf.get_default_graph()
    with graph.as_default():
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=(input_shape,), name="layer_0"))
        model.add(layers.Dense(16, activation='relu', name="layer_1"))
        model.add(layers.Dense(output_shape, activation='sigmoid', name="layer_2"))

        input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, input_shape])
        output_tensor = model(input_tensor)
        graph.add_to_collection("INPUT", input_tensor)
        graph.add_to_collection("OUTPUT", output_tensor)
    return input_tensor, output_tensor


def save_model(sess: tf.Session, path: str) -> None:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    save_path = saver.save(sess, path)
    print(f"Saved the model in {save_path}")


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_shape", type=int)
    parser.add_argument("-o", "--output_shape", type=int)
    parser.add_argument("-p", "--path", type=str)
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    input_tensor, output_tensor = build_model(args.input_shape, args.output_shape)
    print(f"Finished building the model: Input tensor {input_tensor} and output tensor {output_tensor}.")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    res = sess.run(output_tensor, feed_dict={input_tensor: np.random.rand(1, args.input_shape)})
    print(f"The random output is {res}.")
    save_model(sess, args.path)


main()

import argparse
from typing import Dict, List

import tensorflow.compat.v1 as tf


def load_and_stitch_graph(
    path: str,
    sess: tf.Session,
    input_map: Dict[str, tf.Tensor],
    return_values: List[str],
    scope: str = "",
) -> List[tf.Tensor]:
    """
    :param path: The path of the model to be loaded from.
    :param sess: The tensorflow session for loading a model.
    :param input_map: dict key is the tag name of the tensor for another Tensor as
           dict value stitches to.
    :param return_values: The tag name for the returned tensors.
    :param scope: The name used for loading a model.
    :return:
    """
    # Load the checkpoint in the tmp graph.
    tmp_graph = tf.Graph()
    with tmp_graph.as_default():
        saver = tf.train.import_meta_graph(path + ".meta", import_scope=scope)

    # Structure the input map for stitching the graph when loading.
    input_map_for_import_graph_def = {
        tmp_graph.get_collection(key)[0].name: val
        for key, val in input_map.items()
    }
    return_values_for_import_graph_def: List[tf.Tensor] = [
        tmp_graph.get_collection(val)[0].name
        for val in return_values
    ]

    # Load and stitch.
    with sess.graph.as_default():
        return_tensors = tf.import_graph_def(
            tmp_graph.as_graph_def(),
            input_map=input_map_for_import_graph_def,
            name="",
            return_elements=return_values_for_import_graph_def
        )
        # only restore if there's something to restore
        if saver is not None:
            saver.restore(sess, path)

    return return_tensors


def parse_arguments() -> argparse.ArgumentParser:
    """
    Stitch the model loaded from path1 to the model loaded from path2.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--path1", type=str)
    parser.add_argument("-p2", "--path2", type=str)
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)

    with graph.as_default():
        input1, output1 = load_and_stitch_graph(
            path=args.path1,
            sess=sess,
            scope="foo",
            input_map={},
            return_values=["INPUT", "OUTPUT"]
        )

        out2 = load_and_stitch_graph(
            path=args.path2,
            sess=sess,
            scope="bar",
            input_map={"INPUT": output1},
            return_values=["OUTPUT"]
        )
    sess.run(out2[0], feed_dict={input1: [[1, 2, 3, 4]]})


main()