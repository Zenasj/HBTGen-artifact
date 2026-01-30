from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf

class TestModel(tf.keras.models.Model):
  @tf.function
  def test(self, x):
    return x

test_model = TestModel()
signatures = [test_model.test.get_concrete_function(tf.TensorSpec([None], tf.float32))]

converter = tf.lite.TFLiteConverter.from_concrete_functions(signatures, test_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# This raises "ValueError: Cannot set tensor: Tensor is unallocated. Try calling allocate_tensors() first"
result = interpreter.get_signature_runner()(x=tf.zeros([0], tf.float32))