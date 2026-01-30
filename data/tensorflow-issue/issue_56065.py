import numpy as np
import tensorflow as tf
from tensorflow import keras

def total_variation(im1):
    score = tf.image.total_variation(im1)
    return tf.cast(score, tf.float16)

input1 = tf.keras.Input(shape=(512,512,3), batch_size=1, dtype=tf.float16)
output = total_variation(input1)
matrix1 = np.arange(512*512*3,dtype='float16').reshape(1,512,512,3)
model = tf.keras.Model(inputs=[input1], outputs=output, name='test')
x = model.predict([matrix1])
x # NaN

def total_variation(im1):
    score = tf.image.total_variation(im1)
    return tf.cast(score, tf.float32)

input1 = tf.keras.Input(shape=(512,512,3), batch_size=1, dtype=tf.float32)
output = total_variation(input1)
matrix1 = np.arange(512*512*3,dtype='float32').reshape(1,512,512,3)
model = tf.keras.Model(inputs=[input1], outputs=output, name='test')
x = model.predict([matrix1])
x # array([1.207955e+09], dtype=float32)