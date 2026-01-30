import tensorflow as tf

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=(1, 1,)))

cell = tf.keras.layers.GRUCell(10)

model.add(tf.keras.layers.RNN(cell, unroll=True))
    
model.save("test.tf", save_format='tf') # fails
#model.save("test.h5", save_format='h5') # works

from tensorflow import keras
from tensorflow.keras import layers


def get_model():
    inputs = layers.Input(shape=(20, 80))
    outputs = layers.LSTM(32, return_sequences=True, unroll=True)(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def main():
    model_dir = '/tmp/rnnt_toy/model'
    model = get_model()
    model.save(model_dir)


if __name__ == '__main__':
    main()