import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

#%%
from distutils.version import LooseVersion
import numpy as np
import tensorflow as tf

# disable eager model for tf=2.x
tf.compat.v1.disable_eager_execution()

batch_size = 100
img_h = 32
img_w = 32
img_min = 0
img_max = 1
channels = 3
num_classes = 10

strategy = tf.distribute.MirroredStrategy()
#%%
def download_data():

    # get raw data
    (trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()
    trainX = trainX.astype(np.float32)
    testX  = testX.astype(np.float32)

    # ont-hot
    trainY = tf.keras.utils.to_categorical(trainY, 10)
    testY  = tf.keras.utils.to_categorical(testY , 10)

    # get validation sets
    training_size = 45000
    validX = trainX[training_size:,:]
    validY = trainY[training_size:,:]

    trainX = trainX[:training_size,:]
    trainY = trainY[:training_size,:]

    return trainX, trainY, validX, validY, testX, testY

#%%
class DataGenerator:

    def __init__(self, sess, dataX, dataY, total_len, batch_size):

        super().__init__()

        self.total_len  = total_len
        self.batch_size = batch_size
        self.cleanX = dataX
        self.totalY = dataY
        self.sess = sess
        self.on_epoch_end()

    def __build_pipeline(self, dataX, dataY):

        # create dataset API
        def preprocess_fn(dataX, dataY):
            
            dataX = tf.image.random_flip_left_right(dataX)

            # workaround solution
            if LooseVersion(tf.__version__) < LooseVersion('1.14.0'):
                outputX = dataX
            else:
                outputX = (dataX, dataY)
            return outputX, dataY

        dataset = tf.data.Dataset.from_tensor_slices( (dataX, dataY) )
        dataset = dataset.shuffle(batch_size * 8)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.dataset   = dataset

    def  __len__(self):

        return self.total_len // self.batch_size

    def on_epoch_end(self):

        # run permutation
        rand_idx = np.random.permutation(self.total_len)
        cleanX = self.cleanX[rand_idx]
        totalY = self.totalY[rand_idx]

        self.__build_pipeline(cleanX, totalY)

#%%
# ref: https://keras.io/examples/cifar10_resnet/
def build_clf():
    #with strategy.scope():
    with tf.compat.v1.variable_scope('optimizer'):
        def resnet_layer(inputs,
                        num_filters=16,
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        batch_normalization=True,
                        conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder

            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)

            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = tf.keras.layers.Conv2D(num_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = tf.keras.layers.BatchNormalization()(x)
                if activation is not None:
                    x = tf.keras.layers.Activation(activation)(x)
            else:
                if batch_normalization:
                    x = tf.keras.layers.BatchNormalization()(x)
                if activation is not None:
                    x = tf.keras.layers.Activation(activation)(x)
                x = conv(x)
            return x

        def cw_loss(y_true, y_pred):
            label_mask  = label_ref
            pre_softmax = x
            if LooseVersion(tf.__version__) < LooseVersion('1.14.0'):
                correct_logit = tf.reduce_sum(label_mask * pre_softmax, axis=1, keep_dims=True)
            else:
                correct_logit = tf.reduce_sum(label_mask * pre_softmax, axis=1, keepdims=True)
            distance = tf.nn.relu( pre_softmax - correct_logit + (1-label_mask) * 10)
            inactivate = tf.cast( tf.less_equal(distance, 1e-9), dtype=tf.float32)
            weight = tf.keras.layers.Activation('softmax')(-1e9*inactivate + distance)
            loss = tf.reduce_sum((1-label_mask) * distance * weight, axis=1)
            loss = tf.math.reduce_mean(loss)
            return loss

        # set model's parameters (depth = n * 6 + 2)
        n = 8
        num_filters = 16

        clf_input = tf.keras.layers.Input(shape=(img_h, img_w, channels), name="model/input")
        label_ref = tf.keras.layers.Input(shape=(num_classes,), name='label_ref')
        input_list = [clf_input, label_ref]

        x = resnet_layer(inputs=clf_input)
        for stack in range(3):
            for res_block in range(n):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                num_filters=num_filters,
                                strides=strides)
                y = resnet_layer(inputs=y,
                                num_filters=num_filters,
                                activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                    num_filters=num_filters,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                x = tf.keras.layers.Add()([x, y])
                x = tf.keras.layers.Activation('relu')(x)
            num_filters *= 2

        x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(num_classes , kernel_initializer='he_normal', activation=None)(x)
        y = tf.keras.layers.Activation('softmax')(x)

        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        clf_model = tf.keras.models.Model(inputs=input_list, outputs=y, name='clf_model')
        clf_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', cw_loss])
    clf_model.summary()

    return clf_model

#%%
if __name__ == '__main__':

    # set GPU
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # reset tf session
    tf.compat.v1.keras.backend.clear_session()
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    tf.compat.v1.keras.backend.set_session(sess)

    # Hyperparameters
    batch_size = 100
    epochs = 1

    # prepare data
    trainX, trainY, validX, validY, testX, testY = download_data()
    train_gen = DataGenerator(sess, trainX, trainY, trainY.shape[0], batch_size)
    valid_gen = DataGenerator(sess, validX, validY, validY.shape[0], batch_size)
    test_gen  = DataGenerator(sess, testX, testY, testY.shape[0], batch_size)

    # build model
    model = build_clf()

    # train model
    model.fit(train_gen.dataset,
                    epochs=epochs,
                    steps_per_epoch = train_gen.__len__(),
                    validation_data=valid_gen.dataset,
                    validation_steps= valid_gen.__len__(),
                    verbose=1)

    # print result
    meta_string = '[Testing]'
    prefix_string = ''
    output = model.evaluate(test_gen.dataset, steps = test_gen.__len__())
    for ii in range( len( model.metrics_names) ):
        meta_string = meta_string + '- {:s}{:s}: {:.3f} '.format(prefix_string, model.metrics_names[ii], output[ii])

    print(meta_string)

#%%
from distutils.version import LooseVersion
import numpy as np
import tensorflow as tf

#tf.compat.v1.disable_eager_execution()

batch_size = 100
img_h = 32
img_w = 32
img_min = 0
img_max = 1
channels = 3
num_classes = 10

strategy = tf.distribute.MirroredStrategy()
#%%
def download_data():

    # get raw data
    (trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()
    trainX = trainX.astype(np.float32)
    testX  = testX.astype(np.float32)

    # ont-hot
    trainY = tf.keras.utils.to_categorical(trainY, 10)
    testY  = tf.keras.utils.to_categorical(testY , 10)

    # get validation sets
    training_size = 45000
    validX = trainX[training_size:,:]
    validY = trainY[training_size:,:]

    trainX = trainX[:training_size,:]
    trainY = trainY[:training_size,:]

    return trainX, trainY, validX, validY, testX, testY

#%%
class DataGenerator:

    def __init__(self, dataX, dataY, total_len, batch_size):

        super().__init__()

        self.total_len  = total_len
        self.batch_size = batch_size
        self.cleanX = dataX
        self.totalY = dataY
        self.on_epoch_end()

    def __build_pipeline(self, dataX, dataY):

        # create dataset API
        def preprocess_fn(dataX, dataY):
            
            dataX = tf.image.random_flip_left_right(dataX)

            # workaround solution
            if LooseVersion(tf.__version__) < LooseVersion('1.14.0'):
                outputX = dataX
            else:
                outputX = (dataX, dataY)
            return outputX, dataY

        dataset = tf.data.Dataset.from_tensor_slices( (dataX, dataY) )
        dataset = dataset.shuffle(batch_size * 8)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.dataset   = dataset

    def  __len__(self):

        return self.total_len // self.batch_size

    def on_epoch_end(self):

        # run permutation
        rand_idx = np.random.permutation(self.total_len)
        cleanX = self.cleanX[rand_idx]
        totalY = self.totalY[rand_idx]

        self.__build_pipeline(cleanX, totalY)

#%%
# ref: https://keras.io/examples/cifar10_resnet/
def build_clf():
    #with strategy.scope():
    if True:
        def resnet_layer(inputs,
                        num_filters=16,
                        kernel_size=3,
                        strides=1,
                        activation='relu',
                        batch_normalization=True,
                        conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder

            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)

            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = tf.keras.layers.Conv2D(num_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))

            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = tf.keras.layers.BatchNormalization()(x)
                if activation is not None:
                    x = tf.keras.layers.Activation(activation)(x)
            else:
                if batch_normalization:
                    x = tf.keras.layers.BatchNormalization()(x)
                if activation is not None:
                    x = tf.keras.layers.Activation(activation)(x)
                x = conv(x)
            return x

        def cw_loss(y_true, y_pred):
            label_mask  = label_ref
            pre_softmax = x
            if LooseVersion(tf.__version__) < LooseVersion('1.14.0'):
                correct_logit = tf.reduce_sum(label_mask * pre_softmax, axis=1, keep_dims=True)
            else:
                correct_logit = tf.reduce_sum(label_mask * pre_softmax, axis=1, keepdims=True)
            distance = tf.nn.relu( pre_softmax - correct_logit + (1-label_mask) * 10)
            inactivate = tf.cast( tf.less_equal(distance, 1e-9), dtype=tf.float32)
            weight = tf.keras.layers.Activation('softmax')(-1e9*inactivate + distance)
            loss = tf.reduce_sum((1-label_mask) * distance * weight, axis=1)
            loss = tf.math.reduce_mean(loss)
            return loss

        # set model's parameters (depth = n * 6 + 2)
        n = 8
        num_filters = 16

        clf_input = tf.keras.layers.Input(shape=(img_h, img_w, channels), name="model/input")
        label_ref = tf.keras.layers.Input(shape=(num_classes,), name='label_ref')
        input_list = [clf_input, label_ref]

        x = resnet_layer(inputs=clf_input)
        for stack in range(3):
            for res_block in range(n):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                num_filters=num_filters,
                                strides=strides)
                y = resnet_layer(inputs=y,
                                num_filters=num_filters,
                                activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                    num_filters=num_filters,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                x = tf.keras.layers.Add()([x, y])
                x = tf.keras.layers.Activation('relu')(x)
            num_filters *= 2

        x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(num_classes , kernel_initializer='he_normal', activation=None)(x)
        y = tf.keras.layers.Activation('softmax')(x)

        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        clf_model = tf.keras.models.Model(inputs=input_list, outputs=y, name='clf_model')
        clf_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', cw_loss])
    clf_model.summary()

    return clf_model

#%%
if __name__ == '__main__':

    # set GPU
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Hyperparameters
    batch_size = 100
    epochs = 1

    # prepare data
    trainX, trainY, validX, validY, testX, testY = download_data()
    train_gen = DataGenerator(trainX, trainY, trainY.shape[0], batch_size)
    valid_gen = DataGenerator(validX, validY, validY.shape[0], batch_size)
    test_gen  = DataGenerator(testX, testY, testY.shape[0], batch_size)

    # build model
    model = build_clf()

    # train model
    model.fit(train_gen.dataset,
                    epochs=epochs,
                    steps_per_epoch = train_gen.__len__(),
                    validation_data=valid_gen.dataset,
                    validation_steps= valid_gen.__len__(),
                    verbose=1)

    # print result
    meta_string = '[Testing]'
    prefix_string = ''
    output = model.evaluate(test_gen.dataset, steps = test_gen.__len__())
    for ii in range( len( model.metrics_names) ):
        meta_string = meta_string + '- {:s}{:s}: {:.3f} '.format(prefix_string, model.metrics_names[ii], output[ii])

    print(meta_string)