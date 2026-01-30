import tensorflow as tf
from tensorflow import keras

HUB_URL = 'https://tfhub.dev/google/speech_embedding/1'
TEST_PATH = '.'

embedding_layer = hub.KerasLayer(HUB_URL, input_shape=(16000,), trainable=False)

model = tf.keras.Sequential([
    embedding_layer
])

model.save(TEST_PATH)
converter = tf.lite.TFLiteConverter.from_saved_model(TEST_PATH)


converter.allow_custom_ops = True
tflite_model = converter.convert()
tflite_model_file = 'converted_model.tflite'

with open(tflite_model_file, "wb") as f:
  f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()