import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

# disable eager model for tf=2.x
if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
    tf.compat.v1.disable_eager_execution()

#%%
from distutils.version import LooseVersion
import numpy as np
import tensorflow as tf

# disable eager model for tf=2.x
if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
    tf.compat.v1.disable_eager_execution()

batch_size = 100
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
def data_pipeline(dataX, dataY):

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
    return dataset

#%%
class custom_model():
    def __init__(self):

        # custom loss
        def cw_loss(y_true, y_pred):

            # workaround solution
            if LooseVersion(tf.__version__) < LooseVersion('1.14.0'):
                label_mask  = y_true
                pre_softmax = clf_out
            else:
                label_mask  = label_ref
                pre_softmax = clf_out                

            # API changed
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

        # API changed
        if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
            model = tf.keras.applications.ResNet50(include_top=True, weights=None, input_shape=(32,32,3), pooling='max', classes=10)
        else:
            model = tf.keras.applications.resnet.ResNet50(include_top=True, weights=None, input_shape=(32,32,3), pooling='max', classes=10)

        clf_input = tf.keras.layers.Input(shape=(32,32,3), name="model/input")
        label_ref = tf.keras.layers.Input(shape=(10,) , name='label_ref')
        clf_out   = model(clf_input)

        # workaround solution
        if LooseVersion(tf.__version__) < LooseVersion('1.14.0'):
            input_list = [clf_input]
        else:
            input_list = [clf_input, label_ref]
    
        clf_model = tf.keras.models.Model(inputs=input_list, outputs=clf_out, name='clf_model')
        clf_model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy', cw_loss])

        self.clf_model = clf_model

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

    # prepare data
    trainX, trainY, validX, validY, testX, testY = download_data()
    train_gen = data_pipeline(trainX, trainY)
    valid_gen = data_pipeline(validX, validY)
    test_gen = data_pipeline(testX, testY)

    # build targeted model
    targeted_model = custom_model()
    model = targeted_model.clf_model
    
    # fit and evalutate
    model.fit(train_gen,
            steps_per_epoch = trainY.shape[0] // batch_size,
            validation_data = valid_gen,
            validation_steps= validY.shape[0] // batch_size,
            epochs=5,
            verbose=2)

    # workaround solution
    if LooseVersion(tf.__version__) < LooseVersion('1.14.0'):
        model.evaluate(testX, testY, verbose=2, batch_size=batch_size)
    else:
        model.evaluate( (testX, testY), testY, verbose=2, batch_size=batch_size)