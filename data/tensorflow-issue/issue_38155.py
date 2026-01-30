from tensorflow import keras
from tensorflow.keras import optimizers

model = CNN()
y = model.predict(inputs)

# for github issue use

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10


if __name__ == '__main__':
    (x_train, y_train), (x_label, y_label) = cifar10.load_data()
    x_train = x_train.astype(np.float32)

    class CNN(tf.keras.Model):
        def __init__(self):
            super(CNN, self).__init__()

            self.conv1 = layers.Conv2D(
                filters=32,
                kernel_size=[5, 5],
                strides=(1, 1),
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01
                ),
                bias_initializer=tf.zeros_initializer(),
                data_format='channels_last'
            )
            # the above output is [n, 32, 32, 32]
            # from the book, the channel number 3 disappears after process
            # 池化输出大小=[（输入大小-卷积核（过滤器）大小）／步长]+1

            self.pool1 = layers.MaxPool2D(
                pool_size=[3, 3], strides=2,
                data_format='channels_last',
                padding='VALID'
            )
            # the output above is [n, 15, 15, 64]

            # self.lrn1 = LRNLayer(
            #     depth_radius=5,
            #     bias=1,
            #     alpha=1,
            #     beta=0.5
            # )
            # during class call, add the lrn layer, output size not change

            self.conv2 = layers.Conv2D(
                filters=64,
                kernel_size=[5, 5],
                strides=(1,1),
                kernel_initializer=tf.random_normal_initializer(
                    mean=0, stddev=0.01
                ),
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                activation=tf.nn.relu,
                data_format='channels_last'
            )
            # padding = same, so output is [n, 15, 15, 64]
            # self.lrn2 = LRNLayer(
            #     depth_radius=5,
            #     bias=1,
            #     alpha=1,
            #     beta=0.5
            # )

            self.pool2 = layers.MaxPool2D(
                pool_size=[3, 3], strides=2,
                data_format='channels_last',
                padding="VALID"
            )
            # output size = [N, 7, 7, 64]

            # self.flatten = layers.Reshape(target_shape=(-1, 7*7*64))

            self.dense1 = layers.Dense(
                units=784,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.05
                ),
                bias_initializer=tf.zeros_initializer()
            )
            # output is [n, 784]
            self.batchNorm = layers.BatchNormalization()

            self.dense2 = layers.Dense(
                units=10,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.05
                ),
                bias_initializer=tf.zeros_initializer()
            )
            # output is 10

        # def build(self, input_shape):
        #         #     super(CNN, self).build(input_shape)

        def call(self, inputs):
            # print(inputs[0, ::])
            print("here inside the model")
            print("inputs.shape = {}".format(inputs.shape)) # (None, 32, 32, 3)
            print("tf.shape = {}".format(tf.shape(inputs)))
            print("type(inputs) = {}".format(type(inputs)))


            x = self.conv1(inputs) # (None, 32, 32, 32)
            print("x.shape = {}".format(x.shape))
            print("tf.shape = {}".format(tf.shape(x)))

            # not useful for this issue.
            # x = self.pool1(x)
            # # x = self.lrn1(x)
            # x = self.conv2(x)
            # # x = self.lrn2(x)
            # x = self.pool2(x)
            # x = tf.reshape(tensor=x, shape=(x.shape[0], -1))
            # x = self.dense1(x)
            # x = self.batchNorm(x)
            # x = self.dense2(x)
            # y = tf.nn.softmax(x)

            return y


    iteration = 500
    batch_size = 500
    learning_rate = 0.001

    model = CNN()
    # model.build(input_shape=(None, 32, 32, 3))
    # # print(model.conv1.input_spec())

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    print("Here try to predict")
    y = model.predict(x_train[:500, ::])

    # for i in range(iteration + 1):
    #     batch = batch(train, batch_size)
    #
    #     print(batch.data.shape, batch.label.shape) # (500, 32, 32, 3) (500,)
    #     print(type(batch.data))
    #
    #    # get some accuracy to examine
    #     if i % 100 == 0:
    #         batch_predict = model.predict(batch.data)
    #         test_predict = model.predict(test.data)
    #
    #         acc_train = accuracy(
    #             prediction=batch_predict, label=batch.label
    #         )
    #
    #         acc_test = accuracy(
    #             prediction=test_predict, label=test.label
    #         )
    #
    #         train_acc_list.append(acc_train)
    #         test_acc_list.append(acc_test)
    #
    #         print("Iter {} of {}: train_acc={}, test_acc={}".format(i, iteration, acc_train, acc_test))
    #
    #     # now do the training
    #     with tf.GradientTape() as tape:
    #         batch_predict = model(batch.data)
    #         loss = tf.keras.losses.sparse_categorical_crossentropy(
    #             y_pred=batch_predict, y_true=batch.label
    #         )
    #         loss = tf.reduce_mean(loss)
    #
    #     train_loss_list.append(loss)
    #
    #     grads = tape.gradient(loss, model.variables)
    #     optimizer.apply_gradients(
    #         grads_and_vars=zip(grads, model.variables)
    #     )

# for github issue use

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

if __name__ == '__main__':

    class CNN2(tf.keras.Model):
        def __init__(self):
            super(CNN2, self).__init__()
            print("initialized successfully")

        def build(self, input_shape):
            super(CNN2, self).build(input_shape)

        def call(self, inputs):
            # print(inputs[0, ::])
            print("here inside the model")
            print("inputs.shape = {}".format(inputs.shape)) # (None, 32, 32, 3)
            print("tf.shape = {}".format(tf.shape(inputs)))
            print("type(inputs) = {}".format(type(inputs)))

            # just return something
            return inputs



    model = CNN2()
    model.build(input_shape=(500, 32, 32, 3))
    # # print(model.conv1.input_spec())

    print("Here try to predict")
    var = tf.zeros(shape=[500, 32, 32, 3])
    print(var[0, 0, 0, :])
    y = model.predict(var)

inputs.shape = (None, 32, 32, 3)

y = model(x)

y = model.predict(x)