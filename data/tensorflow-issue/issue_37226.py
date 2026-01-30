import tensorflow as tf

class SomeModule(tf.Module):

  @tf.function(input_signature=[
    tf.RaggedTensorSpec(shape=[None, None], dtype=tf.string)
  ])
  def return_ragged_inputs(self, inputs: tf.RaggedTensor):
    return inputs

ragged_inputs = tf.ragged.constant([["foo"], ["bar"], ["foo", "bar"]], dtype=tf.string)

some_module = SomeModule()
return_ragged_inputs_result = some_module.return_ragged_inputs(ragged_inputs)
print(f"ragged_inputs => {ragged_inputs}\n")
print("We can use the method with ragged inputs:")
print(f"return_ragged_inputs => {return_ragged_inputs_result}\n")
print("Saving some_module...")
tf.saved_model.save(some_module, export_dir="foobar")
some_module = tf.saved_model.load("foobar")
print(f"\nWe can pass and return ragged inputs even by saving and reloading the module, if we use it 'eagerly':")
print(some_module.return_ragged_inputs(ragged_inputs))
print("\nBut the graph signature tell us that our exported signature expects tf.Tensor values instead:")
exported_model_graph_inputs = some_module.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs
exported_model_graph_outputs = some_module.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs
print(f"Expected inputs  : {exported_model_graph_inputs}")
print(f"Expected outputs : {exported_model_graph_outputs}")
print("\nThis means we couldn't use the method as we would expect in (e.g.) TensorFlow Serving.")