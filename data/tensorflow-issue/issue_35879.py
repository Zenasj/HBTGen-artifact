import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tensorflow.keras.utils import Sequence
import tensorflow as tf
import numpy as  np

class DataGenerator(Sequence):
    def __init__(self):
        self.batch_size = 32
        self.output_shape = (6, 12)

    def __len__(self):
        return 128

    def __getitem__(self, index):
        X = np.random.uniform(-1, 1, (self.batch_size, *self.output_shape))
        y = np.random.uniform(-1, 1, (self.batch_size, 1))
        return (X, y)

def build_model():
    model = tf.keras.Sequential(name='hello')
    model.add(tf.keras.layers.Flatten(input_shape=(6, 12)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MeanSquaredError())
    return model

if __name__== "__main__":

    gen = DataGenerator()
    val_gen = DataGenerator()

    model = build_model()
    model.fit(x = gen, 
        validation_data = val_gen,
        epochs = 8,
        workers = 4,
        use_multiprocessing = True,
        shuffle=True)

keras.Sequence