import random
from tensorflow import keras

import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.a = tf.Variable(initial_value=tf.random.normal([5]), trainable=False, name='a')
        self.call = tf.function(func=self.call, input_signature=[
            tf.TensorSpec(shape=[None, 5], dtype=tf.float32)
        ])
        self.prod = tf.function(func=self.prod, input_signature=[
            tf.TensorSpec(shape=[None, 5], dtype=tf.float32)
        ])
        self.set = tf.function(func=self.set, input_signature=[
            tf.TensorSpec(shape=[5], dtype=tf.float32)
        ])
        self.build(input_shape=tf.TensorShape(dims=[None, 5]))

    def call(self, inputs):
        print(f'Tracing call with inputs={inputs}')
        return self.a + inputs

    def prod(self, inputs):
        print(f'Tracing prod with inputs={inputs}')
        return self.a * inputs

    def set(self, value):
        print(f'Tracing set with inputs={value}')
        self.a.assign(value)


if __name__ == '__main__':
    model = Model()
    print(model(tf.zeros(shape=[2, 5])))
    print(model.prod(tf.ones(shape=[2, 5])))
    model.set(tf.ones(shape=[5]))
    print(model(tf.zeros(shape=[2, 5])))
    print(model.prod(tf.ones(shape=[2, 5])))
    print(model.weights)
    model.save_weights('/tmp/a.h5')
    del model
    model = Model()
    model.load_weights('/tmp/a.h5')
    print(model(tf.zeros(shape=[2, 5])))
    print(model.prod(tf.ones(shape=[2, 5])))