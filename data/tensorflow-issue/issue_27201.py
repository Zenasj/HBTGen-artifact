from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.framework import graph_util
import graph_util_fixed as graph_util  # using file with fix cebce4a
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import gfile

model = keras.models.Sequential()
model.add(keras.layers.Embedding(0x1000, output_dim=128))
model.add(keras.layers.LSTM(512))
model.add(keras.layers.Dense(1, activation="sigmoid"))

sess = keras.backend.get_session()
output_node_names = [node.op.name for node in model.outputs]
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)
graph_io.write_graph(constant_graph, ".", "test.pb", as_text=False)

with tf.Session() as sess:
    with gfile.FastGFile("test.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")