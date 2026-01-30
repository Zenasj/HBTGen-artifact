from tensorflow import keras
from tensorflow.keras import layers

script
import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(3, activation='sigmoid'))
compiler_params = {
    "optimizer": {
        'class_name': 'Nadam',
        'config': {
            'lr': 0.0001
        }
    }
}
model.compile(**compiler_params)