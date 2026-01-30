import random
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def model(q, v):
    x = layers.MultiHeadAttention(num_heads=2, key_dim=2)(q, v)
    return x

def representative_dataset():
    for _ in range(100):
        q = np.log(np.random.random((8, 16)))
        v = np.log(np.random.random((4, 16)))

        yield [q.astype(np.float32), v.astype(np.float32)]

target = tf.keras.Input(shape=[8, 16])
source = tf.keras.Input(shape=[4, 16])
out = model(target, source)
model = tf.keras.Model(inputs=(target, source), outputs=out)
model.summary()

model.save('MultiHeadAttention.h5')

run_model = tf.function(model)
# let's fix the input size.
concrete_func = run_model.get_concrete_function((
    tf.TensorSpec([1, 8, 16], model.inputs[0].dtype), tf.TensorSpec([1, 4, 16], model.inputs[0].dtype)))

# model directory.
MODEL_DIR = "MultiHeadAttention"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()
# Save the model.
with open('MultiHeadAttention.tflite', 'wb') as f:
    f.write(tflite_model)

print(f'Created model: MultiHeadAttention.tflite')