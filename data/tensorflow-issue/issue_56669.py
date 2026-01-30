import tensorflow as tf
from tensorflow import keras

class Model(tf.keras.Model):
   def __init__(self, name: str = None):
        super().__init__()
        self.dnn_model = self
        print('here', self.trainable_variables)  # here

test = Model("test")