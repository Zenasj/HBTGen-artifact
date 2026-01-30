import random

import tensorflow as tf

class Dense(tf.Module):
    def __init__(self, input_dim, output_size, name=None):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([input_dim, output_size]), name='w')
        self.b = tf.Variable(tf.zeros([output_size]), name='b')
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

test = Dense(2,4)
output = test([[7.0,3]])
# print(output)
# print(test.trainable_variables)

def loss_fn():
    y_true = tf.ones([1,4])

    loss = tf.reduce_mean(tf.square(y_true - output))
    return loss

for i in range(10):
    train_op = tf.compat.v1.train.AdamOptimizer(0.4).minimize(loss_fn, var_list=test.trainable_variables)
    print(train_op)

import tensorflow as tf

class Dense(tf.Module):
    def __init__(self, input_dim, output_size, name=None):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([input_dim, output_size]), name='w')
        self.b = tf.Variable(tf.zeros([output_size]), name='b')


    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

model = Dense(2,4)

class Test(object):
    def __init__(self):
        self.output = model([[7.0, 3]])

    def loss_fn(self):
        y_true = tf.ones([1,4])

        # loss = tf.reduce_mean(tf.square(y_true - self.output)) # error

        output2 = model([[7.0, 3]])
        loss = tf.reduce_mean(tf.square(y_true - output2))

        print(loss)
        return loss

    def run(self):
        for i in range(20):
            train_op = tf.compat.v1.train.AdamOptimizer(0.4).minimize(self.loss_fn)

Test().run()

import tensorflow as tf

class Dense(tf.Module):
    def __init__(self, input_dim, output_size, name=None):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([input_dim, output_size]), name='w')
        self.b = tf.Variable(tf.zeros([output_size]), name='b')


    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

model = Dense(2,4)

class Test(object):
    def __init__(self):
        self.output = model([[7.0, 3]])

    def loss_fn(self):
        y_true = tf.ones([1,4])

        loss = tf.reduce_mean(tf.square(y_true - self.output)) 

        print(loss)
        return loss

    def run(self):
        for i in range(20):
            train_op = tf.compat.v1.train.AdamOptimizer(0.4).minimize(self.loss_fn)
            self.output = model([[7.0, 3]])

Test().run()