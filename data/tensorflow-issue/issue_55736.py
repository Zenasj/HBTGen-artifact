import tensorflow as tf
from tensorflow.python.grappler import tf_optimizer
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import wrap_function
from tensorflow.python.eager import context
from tensorflow.python.framework import convert_to_constants


class MyModel(tf.Module):
  """Simple Fibonacci model.
  To get this bug, I need a loop and two tensor arrays of the same dimensions,
  using a different dtype so the TensorListReserve nodes don't get merged.
  """
  @tf.function
  def __call__(self, n):
    ta = tf.TensorArray(tf.float32, size=n)
    tb = tf.TensorArray(tf.int32, size=n)
    ta = ta.write(0, 0.)
    ta = ta.write(1, 1.)
    tb = tb.write(0, 0)
    tb = tb.write(1, 1)

    for i in range(2, n):
      ta = ta.write(i, ta.read(i - 1) + ta.read(i - 2))
      tb = tb.write(i, tb.read(i - 1) + tb.read(i - 2))

    return ta.stack() + tf.cast(tb.stack(), dtype=tf.float32)


def run_grappler(func, graph_def):
  meta_graph = export_meta_graph(graph_def=graph_def, graph=func.graph)

  # Add a collection 'train_op' so that Grappler knows the outputs.
  fetch_collection = meta_graph_pb2.CollectionDef()
  for array in func.inputs + func.outputs:
    fetch_collection.node_list.value.append(array.name)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

  # Configure Grappler to execute one pass of common subgraph elimination.
  config = config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  rewrite_options.optimizers.extend([
    "common_subgraph_elimination"
  ])
  rewrite_options.meta_optimizer_iterations = 1
  return tf_optimizer.OptimizeGraph(config, meta_graph)

my_model = MyModel()
func = my_model.__call__.get_concrete_function(
    tf.TensorSpec([], tf.int32))

# Freeze the function
frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)

# Run common subgraph elimination
graph_def = frozen_func.graph.as_graph_def()
new_graph_def = run_grappler(func, graph_def)

# Remove the old functions from the context
for f in new_graph_def.library.function:
  while context.context().has_function(f.signature.name):
    context.context().remove_function(f.signature.name)

# Reconstruct a function from the graph definition
new_func = wrap_function.function_from_graph_def(
    new_graph_def,
    [tensor.name for tensor in frozen_func.inputs],
    [tensor.name for tensor in frozen_func.outputs])