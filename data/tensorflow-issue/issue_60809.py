import math

import tensorflow as tf


class Erfer(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, ], dtype=tf.float32)])
    def erf(self, x):
        return tf.math.erf(x)


model = Erfer()
tf.saved_model.save(model, 'saved_model')

import tensorflow as tf

# Path to the saved model directory
saved_model_dir = 'saved_model'

# Convert the saved model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

# Save the TFLite model to a file
tflite_model_file = 'erf.tflite'
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)

model = Model()
concrete_function = model.erf.get_concrete_function()
tf.io.write_graph(concrete_function.graph, ".", "model_file_name", True)