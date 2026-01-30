import random

from __future__ import division
import numpy as np

import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
import os
import sys
from argparse import ArgumentParser

BATCH_SIZE = 1
IMAGE_SIZE = 640


class model_infer:

    def __init__(self):
        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument('-g', "--input-graph",
                                help='Specify the input graph.',
                                dest='input_graph')

        # parse the arguments
        self.args = arg_parser.parse_args()

        self.input_layer = 'serving_default_input_tensor'
        self.output_layers = 'StatefulPartitionedCall'
        self.output_node_index = ['0', '1', '2', '3', '4']
        self.load_graph()

        self.input_tensor = self.infer_graph.get_tensor_by_name(
            self.input_layer + ":0")
        self.output_tensors = [self.infer_graph.get_tensor_by_name(self.output_layers + ":" + index) for index in self.output_node_index]


    def load_graph(self):
        print('load graph from: ' + self.args.input_graph)

        self.infer_graph = tf.Graph()
        with self.infer_graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.FastGFile(self.args.input_graph, 'rb') as input_file:
                input_graph_content = input_file.read()
                graph_def.ParseFromString(input_graph_content)
            output_graph = optimize_for_inference(graph_def, [self.input_layer],
                                                  [self.output_layers], dtypes.uint8.as_datatype_enum, False)
            tf.import_graph_def(output_graph, name='')
        print('----------------------load graph: success------------------------', flush=True)

    def run_benchmark(self):
        with tf.compat.v1.Session(graph=self.infer_graph) as sess:
            input_images = np.random.normal(size=[BATCH_SIZE,
                    IMAGE_SIZE, IMAGE_SIZE, 3])
            _ = sess.run(self.output_tensors, {
                                 self.input_tensor: input_images})



if __name__ == "__main__":
    infer = model_infer()
    infer.run_benchmark()