from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow import keras

# For our use case this would be a quantizer with a custom gradient
def projection(x):
    return 2 * x

class CustomDense(keras.layers.Dense):
    def call(self, inputs):
        original_kernel = self.kernel
        self.kernel = projection(self.kernel)
        outputs = super().call(inputs)
        self.kernel = original_kernel  # reset the original kernel to make this work in eager mode
        return outputs

model = keras.models.Sequential([CustomDense(32, input_shape=(32,))])

assert model.layers[0].kernel in model.layers[0].trainable_weights

from tensorflow import keras
from tensorflow.python.training.tracking.base import (
         no_automatic_dependency_tracking_scope,
     )

def projection(x):
    return 2 * x

class CustomDense(keras.layers.Dense):
    def call(self, inputs):
        original_kernel = self.kernel
        with no_automatic_dependency_tracking_scope(self):
            self.kernel = projection(self.kernel)
        outputs = super().call(inputs)
        with no_automatic_dependency_tracking_scope(self):
            self.kernel = original_kernel
        return outputs

model = keras.models.Sequential([CustomDense(32, input_shape=(32,))])

assert model.layers[0].kernel in model.layers[0].trainable_weights