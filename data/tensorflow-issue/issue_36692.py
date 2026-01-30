import random
from tensorflow.keras import layers

import numpy
import tensorflow
import tensorflow.keras
import tensorflow.lite

def representative_dataset_gen():
    yield [numpy.random.uniform(low=-1, high=1, size=(1,28,28,16)).astype(numpy.float32)]


model=tensorflow.keras.Sequential()
model.add(
    tensorflow.keras.layers.Conv2D(
        filters=16, kernel_size=7, dilation_rate=(2,2), input_shape=(28,28,16),
        use_bias=True, bias_initializer='ones'
    )
)

converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset_gen
converter.experimental_new_converter = False

tflite_model = converter.convert()