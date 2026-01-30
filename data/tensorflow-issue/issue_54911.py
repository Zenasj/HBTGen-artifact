import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np


class StdConv(tf.keras.layers.Conv2D):
    """Weight Standardization Conv2D.

    See https://arxiv.org/pdf/1903.10520v1.pdf.

    """

    def _standardize_wts(self, wts):
        wts_mean = tf.math.reduce_mean(wts, axis=(0, 1, 2), keepdims=True)
        wts_var = tf.math.reduce_variance(wts, axis=(0, 1, 2), keepdims=True)
        return (wts - wts_mean) / tf.math.sqrt(wts_var + 1e-5)

    def call(self, inputs):
        standardized_wts = self._standardize_wts(self.kernel)
        self.kernel.assign(standardized_wts)
        return super().call(inputs)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.root = StdConv(
            64,
            kernel_size=7,
            padding="same",
            name="conv_root",
        )

    def call(self, inputs, training=True):
        return self.root(inputs, training=training)


model = MyModel()
inp = np.ndarray((1, 256, 256, 3))
op = model(inp, training=False)
print("op shape: ", op.shape)

# save tf model
tf.keras.models.save_model(model, "stdconvmodel")


# convert to trt
params = tf.experimental.tensorrt.ConversionParams(precision_mode="FP32")

converter = tf.experimental.tensorrt.Converter(
    input_saved_model_dir="stdconvmodel", conversion_params=params
)

print("\nconverter.convert...")
converter.convert()
print("\nconverter.save...")
converter.save(trt_model)
print("\nsaved!")