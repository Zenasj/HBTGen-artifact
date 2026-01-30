import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CustomAddingWithMasking(tf.keras.layers.Layer):
    def __init__(self, masking_boolean, **kwargs):
        super(CustomAddingWithMasking, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs[0] + inputs[1]
    
    def compute_mask(self, inputs, mask=None):
        return mask