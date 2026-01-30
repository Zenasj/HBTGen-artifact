from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import time


optimizer1 = keras.optimizers.Adadelta()
optimizer2 = tensorflow.train.experimental.enable_mixed_precision_graph_rewrite(optimizer1)
opts_dict = {'fp32': optimizer1, 'mix': optimizer2}
batch_sizes = [256, 512, 1024, 2048, 4096, 8192]
num_classes = 10
epochs = 30

dataset = 'mnist'

for batch_size in batch_sizes:
    for precision in opts_dict:

        optimizer = opts_dict[precision]
        start_time = time.time()

        # the data, split between train and test sets
        if dataset == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            if K.image_data_format() == 'channels_first':
                x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
                x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
                input_shape = (1, 28, 28)
            else:
                x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
                x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
                input_shape = (28, 28, 1)

        if dataset == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            input_shape = x_train.shape[1:]


        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        input = Input(shape=input_shape)
        x = Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape)(input)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=input, outputs=x)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Batch size = %s' % batch_size)
        print('Precision = %s' % precision)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        dur = time.time() - start_time
        print('Run time = %s s' % dur)
        print('================================')

optimizer2 = tensorflow.train.experimental.enable_mixed_precision_graph_rewrite(optimizer1)