from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Flatten, Dense, Input, add, Activation
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

stack_n = 3  # layers = stack_n * 6 + 2
weight_decay = 1e-4
batch_size = 128
iterations = 50000 // batch_size + 1
learning_rate = 1e-1
epoch_num = 200

def Residual_block(inputs, channels, strides=(1, 1)):
    if strides == (1, 1):
        shortcut = inputs
    else:
        shortcut = Conv2D(channels, (1, 1), strides=strides, kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        shortcut = BatchNormalization(momentum=0.9, epsilon=1e-5)(shortcut)
    net = Conv2D(channels, (3, 3), padding='same', strides=strides, kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = Activation('relu')(net)
    net = Conv2D(channels, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(net)
    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = add([net, shortcut])
    net = Activation('relu')(net)
    return net

def ResNet(inputs):
    net = Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = Activation('relu')(net)

    for i in range(stack_n):
        net = Residual_block(net, 16)

    net = Residual_block(net, 32, strides=(2, 2))
    for i in range(stack_n - 1):
        net = Residual_block(net, 32)

    net = Residual_block(net, 64, strides=(2, 2))
    for i in range(stack_n - 1):
        net = Residual_block(net, 64)

    net = AveragePooling2D(8, 8)(net)
    net = Flatten()(net)
    net = Dense(10, activation='softmax')(net)
    return net

def scheduler(epoch):
    if epoch < epoch_num * 0.4:
        return learning_rate
    if epoch < epoch_num * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01

if __name__ == '__main__':
    # load data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.0)
    datagen.fit(train_images)

    # get model
    img_input = Input(shape=(32, 32, 3))
    output = ResNet(img_input)
    model = models.Model(img_input, output)

    # show
    model.summary()

    # train
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    change_lr = LearningRateScheduler(scheduler)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    datagenflow = datagen.flow(train_images, train_labels, batch_size=batch_size)
    model.fit_generator(datagenflow,
                        steps_per_epoch=iterations,
                        epochs=epoch_num,
                        callbacks=[change_lr],
                        validation_data=(test_images, test_labels))