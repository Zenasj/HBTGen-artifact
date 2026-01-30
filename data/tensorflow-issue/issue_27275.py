import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


class MyLayer(tfk.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def call(self, inputs, training=None):
        # got: Tensor("keras_learning_phase:0", shape=(), dtype=bool)
        print("layer training arg: ", training)
        
        return inputs


class MyModel(tfk.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = MyLayer()

    def call(self, inputs, training=None):
        # want: Tensor("keras_learning_phase:0", shape=(), dtype=bool)  got: None
        print("model training arg: ", training)
        
        inputs = self.l1(inputs)
        return inputs


if __name__ == '__main__':
    x = np.zeros((1, 2))
    model = MyModel()
    model.compile(loss="mse", optimizer="sgd")
    model.fit(x, x, epochs=1)