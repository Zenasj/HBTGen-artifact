def serialize_concrete_function(concrete_function, node_ids, coder):
  """Build a SavedConcreteFunction."""
  bound_inputs = []
  try:
    # import pdb;pdb.set_trace()
    print('-----------------------------------------------------')
    print("concrete_function:", concrete_function)
    print('node_ids:',node_ids)
    for capture in concrete_function.captured_inputs:
      print('capture:', capture)
      bound_inputs.append(node_ids[capture])
      print("bound_inputs:", bound_inputs)
    print("bound_inputs:",bound_inputs)
    print('-----------------------------------------------------')
  except KeyError:
    raise KeyError(
        "Failed to add concrete function %s to object based saved model as it "
        "captures tensor %s which is unsupported or not reachable from root. "
        "One reason could be that a stateful object or a variable that the "
        "function depends on is not assigned to an attribute of the serialized "
        "trackable object "
        "(see SaveTest.test_captures_unreachable_variable)."
        % (concrete_function.name, capture))
  concrete_function_proto = saved_object_graph_pb2.SavedConcreteFunction()
  structured_outputs = func_graph_module.convert_structure_to_signature(
      concrete_function.structured_outputs)
  concrete_function_proto.canonicalized_input_signature.CopyFrom(
      coder.encode_structure(concrete_function.structured_input_signature))
  concrete_function_proto.output_signature.CopyFrom(
      coder.encode_structure(structured_outputs))
  concrete_function_proto.bound_inputs.extend(bound_inputs)
  return concrete_function_proto

class ExampleModel(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
  def capture_fn(self, x):
    if not hasattr(self, 'weight'):
      self.weight = tf.Variable(5.0, name='weight')
    self.weight.assign_add(x * self.weight)
    return self.weight

  @tf.function
  def polymorphic_fn(self, x):
    return tf.constant(3.0) * x

model = ExampleModel()
# model.polymorphic_fn(tf.constant(4.0))
# model.polymorphic_fn(tf.constant([1.0, 2.0, 3.0]))
tf.saved_model.save(
    model, "/tmp/example-model", signatures={'capture_fn': model.capture_fn})

class ExampleModel(tf.Module):
  # def __init__(self):
  #   self.polymorphic_fn(tf.constant(4.0))

  @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
  def capture_fn(self, x):
    if not hasattr(self, 'weight'):
      self.weight = tf.Variable(5.0, name='weight')
      self.bias = tf.Variable(5.0, name='bias')
    self.weight.assign_add(x * self.weight)
    self.bias.assign_add(x * self.weight)
    return self.weight

  @tf.function
  def polymorphic_fn(self, x):
    return tf.constant(3.0) * x

model = ExampleModel()
# model.polymorphic_fn(tf.constant(4.0))
# model.polymorphic_fn(tf.constant(4.0))
model.polymorphic_fn(tf.constant([1.0, 2.0, 3.0]))
tf.saved_model.save(
    model, "/tmp/example-model", signatures={'capture_fn': model.capture_fn})

import os
import datetime
import tensorflow as tf


class ExampleModel(tf.Module):
  def __init__(self):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/test11/%s' % stamp
    self.summary_writer = tf.summary.create_file_writer(logdir)

  @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
  def capture_fn(self, x):
    if not hasattr(self, 'weight'):
      self.weight = tf.Variable(5.0, name='weight')
    self.weight.assign_add(x * self.weight)

    # no error if these two lines are commented
    with self.summary_writer.as_default():
      tf.summary.scalar('loss', tf.constant(0, dtype=tf.int64), step=tf.constant(0, dtype=tf.int64))

    return self.weight

  @tf.function
  def polymorphic_fn(self, x):
    return tf.constant(3.0) * x

model = ExampleModel()
model.polymorphic_fn(tf.constant([1.0, 2.0, 3.0]))
tf.saved_model.save(
    model, "/tmp/example-model", signatures={'capture_fn': model.capture_fn})