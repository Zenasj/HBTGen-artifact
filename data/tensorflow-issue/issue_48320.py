import math

import tensorflow as tf
import numpy as np

print('tf-version:', tf.__version__)

class Test(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def stats(self, sample_input):
        return {
            'mean': tf.math.reduce_mean(sample_input, axis=0),
            'std': tf.math.reduce_std(sample_input, axis=0)
        }

model = Test()

# Convert the SavedModel using TFLiteConverter
SAVED_MODEL_PATH = 'content/saved_models/coding'
tf.saved_model.save(
    model, SAVED_MODEL_PATH,
    signatures={
      'stats': model.stats.get_concrete_function()
    })
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content=tflite_model)
signatures = interpreter.get_signature_list()
print("[SavedModel] signatures: ", signatures)

# Convert the concrete functions using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [model.stats.get_concrete_function()], model)
tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content=tflite_model)
signatures = interpreter.get_signature_list()
print("[Interpreter] signatures: ", signatures)