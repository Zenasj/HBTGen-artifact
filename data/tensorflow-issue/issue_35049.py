import random
from tensorflow import keras

#%%
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#tf.compat.v1.disable_v2_behavior()

import numpy as np

batch_size = 100

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

def data_pipeline(dataX, dataY):

    # create dataset API
    def preprocess_fn(dataX, dataY):
        
        dataX = tf.image.random_flip_left_right(dataX)
        return dataX, dataY

    dataset = tf.data.Dataset.from_tensor_slices( (dataX, dataY) )
    dataset = dataset.shuffle(batch_size * 8)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

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

    # prepare data
    trainX, trainY, validX, validY, testX, testY = download_data()
    train_gen = data_pipeline(trainX, trainY)
    valid_gen = data_pipeline(validX, validY)
    test_gen = data_pipeline(testX, testY)

    # build targeted model
    model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None, input_shape=(32,32,3), pooling='max', classes=10)
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    # fit and evalutate
    num_epoch = 20
    for ii in range(num_epoch):
        model.fit(train_gen,
                steps_per_epoch = trainY.shape[0] // batch_size,
                validation_data = valid_gen,
                validation_steps= validY.shape[0] // batch_size,
                epochs=1,
                verbose=2)
        model.evaluate(testX, testY, verbose=2, batch_size=batch_size)

        # update trainX and re-generate train_gen
        trainX = trainX + 0
        train_gen = data_pipeline(trainX, trainY)

#%%
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#tf.compat.v1.disable_v2_behavior()

import numpy as np

batch_size = 100

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

class DataGenerator():

    def __init__(self, sess, dataX, dataY, batch_size):

        self.batch_size = batch_size
        self.sess = sess
        self.rawX = dataX
        self.rawY = dataY

        # create dataset API
        def preprocess_fn(dataX, dataY):

            dataX = tf.image.random_flip_left_right(dataX)
            return dataX, dataY

        tf_dataX = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32, 3])
        tf_dataY = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

        dataset = tf.data.Dataset.from_tensor_slices( (tf_dataX, tf_dataY) )
        dataset = dataset.shuffle(self.batch_size * 8)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        tf_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
        tf_next = tf_iter.get_next()

        self.tf_dataX = tf_dataX
        self.tf_dataY = tf_dataY
        self.dataset   = dataset
        self.tf_iter   = tf_iter
        self.tf_next   = tf_next

        self.on_epoch_end()

    def  __len__(self):

        return self.rand_idx.shape[0] // self.batch_size

    def __getitem__(self, index):

        return self.sess.run(self.tf_next)

    def on_epoch_end(self):

        dataX  = self.rawX.copy()
        dataY  = self.rawY.copy()

        # run permutation
        total_len = dataY.shape[0]
        self.rand_idx = np.random.permutation(total_len)
        dataX = dataX[self.rand_idx,:]
        dataY = dataY[self.rand_idx,:]

        self.sess.run(self.tf_iter.initializer,
                    feed_dict={ self.tf_dataX: dataX,
                                self.tf_dataY: dataY})

def data_pipeline(dataX, dataY):

    # create dataset API
    def preprocess_fn(dataX, dataY):
        
        dataX = tf.image.random_flip_left_right(dataX)
        return dataX, dataY

    dataset = tf.data.Dataset.from_tensor_slices( (dataX, dataY) )
    dataset = dataset.shuffle(batch_size * 8)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

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

    # prepare data
    trainX, trainY, validX, validY, testX, testY = download_data()
    train_gen = DataGenerator(sess, trainX, trainY, batch_size)
    valid_gen = DataGenerator(sess, validX, validY, batch_size)
    test_gen = data_pipeline(testX, testY)

    # build targeted model
    model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None, input_shape=(32,32,3), pooling='max', classes=10)
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    # fit and evalutate
    num_epoch = 20
    for ii in range(num_epoch):
        model.fit(train_gen.tf_iter,
                steps_per_epoch = trainY.shape[0] // batch_size,
                validation_data = valid_gen.tf_iter,
                validation_steps= validY.shape[0] // batch_size,
                epochs=1,
                verbose=2)
        model.evaluate(testX, testY, verbose=2, batch_size=batch_size)

        # update trainX and re-generate train_gen
        train_gen.on_epoch_end()

def on_epoch_end(self):

        dataX  = self.rawX.copy()
        dataY  = self.rawY.copy()

        # generate adversarial data
        eps = 8
        grads = model.get_gradients(loss, dataX)
        AdvX = dataX + eps * np.sign(grads)

        # combine data
        dataX = np.vstack([dataX, AdvX])
        dataY = np.vstack([dataY, dataY])

        # run permutation
        total_len = dataY.shape[0]
        self.rand_idx = np.random.permutation(total_len)
        dataX = dataX[self.rand_idx,:]
        dataY = dataY[self.rand_idx,:]

        self.sess.run(self.tf_iter.initializer,
                    feed_dict={ self.tf_dataX: dataX,
                                self.tf_dataY: dataY})