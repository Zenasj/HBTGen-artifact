import tensorflow as tf

class Model(tf.Module):
  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])           # ---> Line 2
  def solve(self, x):
    ...
    ...
    return y


# Save the model in the usual way:
model = Model()
SAVED_MODEL_PATH = 'updated_model'
tf.saved_model.save(
model, SAVED_MODEL_PATH,
signatures={
    'solve': model.solve.get_concrete_function()
})

# The way I am converting the model:
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS,
tf.lite.OpsSet.SELECT_TF_OPS 
]
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS, 
tf.lite.OpsSet.SELECT_TF_OPS
]
converter.inference_input_type = tf.float64    # Added lines
converter.inference_output_type = tf.float64
tflite_model = converter.convert()

converter.inference_input_type = tf.float64 ,
converter.inference_output_type = tf.float64