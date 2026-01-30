from tensorflow import keras

import tensorflow as tf

tf.keras.backend.clear_session()

class Manager(tf.keras.Model):
    def __init__(self):
        super(Manager, self).__init__()

        self.inputs = tf.keras.Input(shape=(None,))

model = Manager()
model.build(input_shape=(16,16)).summary()

tf.keras.Model

build