import tensorflow as tf
from tensorflow import keras

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

model = tf.keras.Model({"input": input_tensor}, {"boxes": boxes, "scores": scores})

print(model.input)
# {'input': <KerasTensor: shape=(None, 32, 32, 128) dtype=float32 (created by layer 'input_4')>}

print(model.output)
# {'boxes': <KerasTensor: shape=(None, 4) dtype=float32 (created by layer 'tf.reshape_6')>,
# 'scores': <KerasTensor: shape=(None,) dtype=float32 (created by layer 'tf.reshape_7')>}

print(interpreter.get_signature_list()["serving_default"])
# {'inputs': ['input_4'], 'outputs': ['tf.reshape_6', 'tf.reshape_7']}