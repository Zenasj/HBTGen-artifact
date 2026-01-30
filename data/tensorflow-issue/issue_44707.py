import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

input_shape = [4, 2]
input1 = tf.keras.Input(shape=input_shape, batch_size=1)
input2 = tf.keras.Input(shape=input_shape, batch_size=1)
output = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(
                                   x[0][0], x[1][0], transpose_b=True)
                                )([input1, input2])
model = tf.keras.Model(inputs=[input1, input2], outputs=output)


def get_rand_date():
    return np.random.rand(1, *input_shape).astype(np.float32)

def representative_data_gen():
    yield [get_rand_date(), get_rand_date()]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.representative_dataset = representative_data_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
interpreter.set_tensor(interpreter.get_input_details()[0]["index"],
                       get_rand_date())
interpreter.set_tensor(interpreter.get_input_details()[1]["index"],
                       get_rand_date())
interpreter.invoke()