from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Add


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.bias = self.add_weight(shape=(10, 64, 64, 256))
        self.add = Add()

    def call(self, x):
        return self.add([x, self.bias])


if __name__ == '__main__':
    mm = MyModel()

    x = Input(shape=(64, 64, 256), batch_size=10, name='Input')
    m = Model(inputs=[x], outputs=mm.call(x))
    tf.keras.utils.plot_model(m)