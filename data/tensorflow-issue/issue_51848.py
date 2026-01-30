import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_apply = tfmot.quantization.keras.quantize_apply



def get_model(quantization_aware: bool = False):
    input_embeddings = tf.keras.Input(shape=(512,), dtype=tf.float32, name="input_embeddings")
    is_training = tf.keras.Input(shape=(), dtype=tf.bool, name="is_training")

    x = tf.keras.layers.Dropout(rate=0.5)(inputs=input_embeddings, training=is_training)

    if quantization_aware:  # apply quantization to dense layer and return quantization-aware model
        out = quantize_annotate_layer(
            tf.keras.layers.Dense(units=512,name="dense")
        )(x)

        model = quantize_apply(
            tf.keras.Model(
                inputs=[input_embeddings, is_training],
                outputs=[x, out],
                name="toy_model",
            )
        )
    else: # return vanilla model
        out = tf.keras.layers.Dense(units=512,name="dense")(x)
        model = tf.keras.Model(
            inputs=[input_embeddings, is_training],
            outputs=[x, out],
            name="toy_model",
        )

    return model

def convert_to_tflite(saved_model_path, output_model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_quant_model = converter.convert()

    with open(output_model_path, "wb") as fh:
        fh.write(tflite_quant_model)



model = get_model(quantization_aware=False)  # set to False will raise the error. Set to True the code runs successfully.
input_data = np.random.rand(16, 512)
embeddings, out = model([input_data, True])

model.save("toy_model")

convert_to_tflite("toy_model", "toy_model.tflite")
interpreter = tf.lite.Interpreter("toy_model.tflite")