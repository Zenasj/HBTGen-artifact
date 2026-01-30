import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TestLSTMCell(tf.keras.layers.LSTMCell):
    def __init__(self, units, **kwargs):
        super(TestLSTMCell, self).__init__(units, **kwargs)