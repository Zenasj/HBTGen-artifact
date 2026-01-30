import random
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(256, name='LSTM', return_sequences=False, use_bias=False,  input_shape=(256,1)))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(1, activation='linear', name='output'))
    return model



if __name__ == "__main__":
    #load lstm model
    input_shape = (None, 256)
    model = lstm_model(input_shape)

    x=np.random.rand(1000, 256, 1)
    y = np.random.rand(1000, 1)
    print(x.shape)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x=x, y=y, epochs=2, validation_split=0.2)

    #freeze the session and save it to a pd.file
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, './', './model.pb', as_text=False)

    #load the pb file into graph
    frozen_model_filename = './model.pb'
    graph = load_graph(frozen_model_filename)