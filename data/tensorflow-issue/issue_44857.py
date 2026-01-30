import tensorflow as tf
from tensorflow import keras

model = DenseNet121(input_shape=(810, 1440, 3),
                                         include_top=False,
                                         weights='imagenet'
                                         )

depth = model.get_output_shape_at(0)[-1]

model(tf.keras.Input((810, 1440, 3)))
depth = model.get_output_shape_at(0)