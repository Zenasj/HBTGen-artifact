from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

class Test_Layer(tf.keras.layers.Layer):

    def __init__(self):
        super(Test_Layer, self).__init__()

    @tf.function
    def call(self, x):
        result = tf.TensorArray(tf.int32, size=x.shape[0])
        for i in tf.range(x.shape[0]):
            if x[i] > 0:
                result = result.write(i, x[i] ** 2)
            else:
                result = result.write(i, x[i])
        return result.stack()



test_layer = Test_Layer()

out = test_layer(tf.range(-5, 5))
print(out) #works fine:= tf.Tensor([-5 -4 -3 -2 -1  0  1  4  9 16], shape=(10,), dtype=int32)


test_model = tf.keras.models.Sequential([test_layer])
test_model.compile(loss=tf.losses.mse)


out = test_model(tf.range(-5, 5))
print(out) #works fine:= tf.Tensor([-5 -4 -3 -2 -1  0  1  4  9 16], shape=(10,), dtype=int32)


out = test_model.predict(tf.range(-5, 5))
print(out) #ERROR