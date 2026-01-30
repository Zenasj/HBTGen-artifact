from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import keras
import h5py
import keras_efficientnets
from custom_objects import EfficientNetConvInitializer
from custom_objects import EfficientNetDenseInitializer
from custom_objects import Swish, DropConnect
if __name__ == "__main__":
    debugger = EfficientNetConvInitializer()
    converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file("efficient_net_v1_190925.h5")
    converter.optimizations = [tf.contrib.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    open("efficient_net_wrap_finger.tflite", "wb").write(tflite_model)

def _hard_swish(self, x):
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

get_custom_objects().update({
    'EfficientNetConvInitializer': EfficientNetConvInitializer,
    'EfficientNetDenseInitializer': EfficientNetDenseInitializer,
    'DropConnect': DropConnect,
    'Swish': Swish,
})

import functools
import keras
import os
from efficientnet import model
from tensorflow.python.keras.utils import CustomObjectScope, get_custom_objects

def inject_keras_modules(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = keras.backend
        kwargs['layers'] = keras.layers
        kwargs['models'] = keras.models
        kwargs['utils'] = keras.utils
        return func(*args, **kwargs)

    return wrapper
def init_keras_custom_objects():
    custom_objects = {
        'swish': inject_keras_modules(model.get_swish)(),
        'FixedDropout': inject_keras_modules(model.get_dropout)()
    }

    get_custom_objects().update(custom_objects)

init_keras_custom_objects()
keras_model_path = './some_model.h5'
save_model = tf.keras.models.load_model(keras_model_path)
export_dir='save'
tf.saved_model.save(save_model, export_dir)
new_model = tf.saved_model.load(export_dir)

IMAGE_WIDTH = 64 # example
with CustomObjectScope({'swish': inject_keras_modules(model.get_swish)(),
                        'FixedDropout': inject_keras_modules(model.get_dropout)()}):
    concrete_func = new_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([1, IMAGE_WIDTH, IMAGE_WIDTH, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

concrete_func = new_model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, IMAGE_WIDTH, IMAGE_WIDTH, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

MODEL_OUTPUT_PATH = "efficient_net_b0.tflite"
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.allow_custom_ops = True
tflite_model = converter.convert()
open(MODEL_OUTPUT_PATH, "wb").write(tflite_model)