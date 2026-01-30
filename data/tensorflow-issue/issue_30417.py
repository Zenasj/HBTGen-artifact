from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np

from tensorflow import keras

# Comment this line out to make code complete successfully
keras.mixed_precision.experimental.set_policy('infer_float32_vars')

class MyLayer(keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight(shape=(16, 16))

    def call(self, inputs, **kwargs):
        w = self.w
        # Uncomment this workaround line below to make it work with mixed-precision ON
        # w = keras.backend.cast(w, dtype=w.dtype)
        return keras.backend.dot(inputs, w[:16, :16])

input = keras.layers.Input(shape=(16, ))
output = MyLayer()(input)
model = keras.models.Model(input, output)

model.predict(np.zeros(shape=(16, 16)))