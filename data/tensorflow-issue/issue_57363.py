from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

class DummyModel(tf.keras.Model):
    feature_extractor = None

    def __init__(self, name="StylePredictionModelDummy"):
        super().__init__(name=name)

        self.feature_extractor = tf.keras.layers.Conv2D(1, 9, 5, padding='same', name="dummy_conv")

    def call(self, inputs, training=None, mask=None):
        x = self.feature_extractor(inputs)
        return x

image_shape = (None, 960//4, 1920//4, 3)

model = DummyModel()
element = tf.convert_to_tensor(np.zeros((1, image_shape[1], image_shape[2], 3)))

# call once to build model
result = model(element)

model.save(filepath="%TEMP%/model", include_optimizer=False, save_format='tf')