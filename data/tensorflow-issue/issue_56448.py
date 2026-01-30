from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow import keras
import h5py
from tensorflow.python.keras.saving import hdf5_format


class CustomLayer(keras.layers.Layer):
    """combine multiple activations weighted by learnable variables"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        return {}

    def build(self, input_shape):
        return

    def call(self, inputs):
        return inputs
    
path = 'test.h5'
    
x = keras.Input((5))
y = CustomLayer()(x)
model = keras.Model(x, y)
model.build(x)
model.save(path)

# this works ok
custom_objects = {'CustomLayer': CustomLayer}
model = keras.models.load_model(path, custom_objects=custom_objects)

# this fails
with h5py.File('test.h5', mode='r') as f:
    saved_model = hdf5_format.load_model_from_hdf5(
        f, custom_objects=custom_objects)