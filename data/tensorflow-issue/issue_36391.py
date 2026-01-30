from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, embedding_dim]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=False,
                        recurrent_activation='sigmoid',
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

embedding_dim = 100
units = 256
vocab_size = 300
batch_size = 32

model = build_model(vocab_size, embedding_dim, units, batch_size)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

from tensorflow.python.keras.saving import saving_utils as _saving_utils
from tensorflow.python.framework import convert_to_constants as _convert_to_constants

tf.keras.backend.set_learning_phase(False)
func = _saving_utils.trace_model_call(model)
concrete_func = func.get_concrete_function()
frozen_func = _convert_to_constants.convert_variables_to_constants_v2(concrete_func)

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction   
frozen_func = convert_variables_to_constants_v2(full_model,lower_control_flow=False)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)