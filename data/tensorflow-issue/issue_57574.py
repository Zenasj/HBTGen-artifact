from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

class MyLayer(Layer):
    def __init__(self):
        super().__init__()
        self.dense = Dense(512)
    def __call__(self, X):
        short = X
        X = self.dense(X)
        X = short + X
        return X


def main():
    model = tf.keras.Sequential([MyLayer()])
    model.build([None, 512])
    model.save("/tmp/my_model")

    tf.keras.models.load_model("/tmp/my_model")

if __name__ == "__main__":
    main()

class MyLayer(Layer):
    def __init__(self):
        super().__init__()
        self.dense = Dense(512)
        self.add = Add()
    def __call__(self, X):
        short = X
        X = self.dense(X)
        X = self.add([short , X])
        return X

def get_model(shape=(None, 512)):
    X_input = Input(shape)

    short = X_input

    X = Dense(512)(X_input)
    X = Add()([short, X])

    return Model(inputs= X_input, outputs=X)