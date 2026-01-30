from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_model_optimization as tfmot

i = tf.keras.layers.Input(shape=(24, 24, 3))
x = tf.keras.layers.Conv2D(10, kernel_size=1, activation='tanh')(i)
model = tf.keras.Model(inputs=i, outputs=x)

quant_aware_model = tfmot.quantization.keras.quantize_model(model)

import tensorflow as tf
import tensorflow_model_optimization as tfmot

i = tf.keras.layers.Input(shape=(24, 24, 3))
x = tf.keras.layers.Conv2D(10, kernel_size=1)(i)
x = tf.nn.tanh(x)
model = tf.keras.Model(inputs=i, outputs=x)

quant_aware_model = tfmot.quantization.keras.quantize_model(model)

tiny_model = ...

idx = -3 # adjust idx to your use case (probably -2, check it by yourself)
model_no_tail = tf.keras.Model(inputs=tiny_model.input, outputs=tiny_model.layers[idx].output)

quant_aware_model = tfmot.quantization.keras.quantize_model(model_no_tail)

def add_tail(quant_aware_model):
    base_model=quant_aware_model
    x = tf.nn.tanh(base_model.output)
    x = tf.quantization.fake_quant_with_min_max_args(x, min=-1, max=1, num_bits=8, narrow_range=False, name=None)
    model=tf.keras.Model(inputs=base_model.input, outputs=x)
    return model

quant_aware_model = add_tail(quant_aware_model)
quant_aware_model.summary()

tflite_models_dir = pathlib.Path(model_path)
tflite_models_dir.mkdir(exist_ok=True, parents=True)
quantized_tflite_encoder = tflite_models_dir/"tiny_model.tflite"

if not os.path.exists(model_path + "tiny_model.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()
    
    # Write it out to a tflite file:
    quantized_tflite_encoder.write_bytes(tflite_model_quant)