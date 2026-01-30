from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dummy_1 = [[[1.1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]],
           [[1.2,2,3,4,5],[2,3,4,5,6.8]],
           [[1.3,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8.9]]]

dummy_2 = [[[1.1,2,3,4,5],[2,3,4,5,6]],
           [[1.1,2,3,4,5],[2,3,4,5,6]],[3,4,5,6,7],
           [[1.3,2,3,4,5],[2,3,4,5,6]]]

dummy_3 = [[[1.5,2,3,4,5],[2,3,4,5,6]],
           [[1.6,2,3,4,5],[2,3,4,5,6]],[3,4,5,6,7],
           [[1.7,2,3,4,5],[2,3,4,5,6]]]

def gen():
    for i in range(len(dummy_1)):
        yield(dummy_1[i],dummy_2[i],dummy_2[i],dummy_3[i])

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true))

class network():
    def __init__(self):
        input_1 = keras.Input(shape=(None,5))
        input_2 = keras.Input(shape=(None,5))
        output_1 = layers.Conv1DTranspose(16, 3, padding='same', activation='relu')(input_1)
        output_2 = layers.Conv1DTranspose(16, 3, padding='same', activation='relu')(input_2)

        self.model = keras.Model(inputs=[input_1, input_2],
                                 outputs=[output_1, output_2])
        
        # compile model
        self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001),
                           loss={"mel_loss":custom_loss, "mag_loss":custom_loss})
    
    def train(self):
        self.dataset = tf.data.Dataset.from_generator(gen, 
                                                      (tf.float32, tf.float32, tf.float32, tf.float32))
        self.dataset.batch(32).repeat()
        self.model.fit(self.dataset,epochs=3)
        #self.model.fit([dummy_1, dummy_2],
        #               [dummy_2, dummy_3],
        #               epochs=3)

net = network()
net.train()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class network():
    def __init__(self):
        input_1 = keras.Input(shape=(None,5))
        input_2 = keras.Input(shape=(None,5))
        output_1 = layers.Conv1DTranspose(16, 3, padding='same', activation='relu')(input_1)
        output_2 = layers.Conv1DTranspose(16, 3, padding='same', activation='relu')(input_2)

        self.model = keras.Model(inputs=[input_1, input_2],
                                 outputs=[output_1, output_2])
    
    def train(self):
        dummy_1 = tf.ragged.constant([[[1.1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]],
                   [[1.2,2,3,4,5],[2,3,4,5,6.8]],
                   [[1.3,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8.9]]])

        dummy_2 = tf.ragged.constant([[[1.1,2,3,4,5],[2,3,4,5,6]],
                   [[1.1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]],
                   [[1.3,2,3,4,5],[2,3,4,5,6]]])

        dummy_3 = tf.ragged.constant([[[1.5,2,3,4,5],[2,3,4,5,6]],
                   [[1.6,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]],
                   [[1.7,2,3,4,5],[2,3,4,5,6]]])
        self.dataset = tf.data.Dataset.from_tensor_slices((dummy_1, dummy_2, dummy_3))
        for dummy_1, dummy_2, dummy_3 in self.dataset.batch(2):
            x, y = self.model((dummy_1, dummy_2), training=True)

net = network()
net.train()