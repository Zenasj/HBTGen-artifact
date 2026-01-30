import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

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

        dataset = tf.data.Dataset.from_tensor_slices( (dataX, dataY) )
        dataset = dataset.shuffle(batch_size * 8)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

class custom_model():
    def __init__(self):

        def Acc():
            acc = tf.keras.metrics.categorical_accuracy(label_ref, clf_out)
            return tf.math.reduce_mean(acc)

        def c_loss():
            loss = tf.keras.losses.categorical_crossentropy(label_ref, clf_out)
            loss = tf.math.reduce_mean(loss)
            return loss

        # create model
        clf_input = tf.keras.layers.Input(shape=(32,32,3), name="model/input")
        model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None, input_tensor=clf_input, pooling='max', classes=10)
        #model = tf.keras.applications.vgg16.VGG16(include_top=True, weights=None, input_tensor=clf_input, pooling='max', classes=10)
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

        label_ref = tf.keras.layers.Input(shape=(10,) , name='label_ref')
        clf_out = model(clf_input)

        # using tf.keras.optimizers.Nadam would get error
        optimizer = tf.keras.optimizers.Nadam(lr=0.0005)
        #optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = optimizer.minimize(c_loss, var_list=[model.trainable_variables])

        self.clf_model = model
        self.clf_input = clf_input
        self.label_ref = label_ref
        self.op_acc = Acc()
        self.c_loss = c_loss()

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
    #model = tf.keras.applications.vgg16.VGG16(include_top=True, weights=None, input_shape=(32,32,3), pooling=None, classes=10)
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    # fit and evalutate
    model.fit(train_gen,
            steps_per_epoch = trainY.shape[0] // batch_size,
            validation_data = valid_gen,
            validation_steps= validY.shape[0] // batch_size,
            epochs=5,
            verbose=2)
    model.evaluate(testX, testY, verbose=2, batch_size=batch_size)

    # create a new model
    print('Make sure that we create a new model.')
    model = custom_model()
    sess.run(tf.compat.v1.global_variables_initializer())
    model.clf_model.evaluate(testX, testY, verbose=2, batch_size=batch_size)

    # train model
    num_epoch = 5
    tf_iter = tf.compat.v1.data.make_initializable_iterator(train_gen)
    tf_next = tf_iter.get_next()
    sess.run(tf_iter.initializer)
    for epoch in range(num_epoch):
        c_loss, acc = 0.0, 0.0
        for ii in range(trainY.shape[0] // batch_size):
            X, Y = sess.run(tf_next)
            [b_c_loss, b_acc, _] = sess.run([model.c_loss, model.op_acc, model.train_op],
                                                feed_dict={ model.clf_input: X,
                                                            model.label_ref: Y,
                                                            tf.keras.backend.learning_phase(): 1})
            c_loss = c_loss + b_c_loss
            acc = acc + b_acc
        
        c_loss = c_loss / (trainY.shape[0] // batch_size)
        acc = acc / (trainY.shape[0] // batch_size)
        print('[Training]Epoch: {:d}/{:d} - loss: {:.3f} - acc: {:.3f}'.format(epoch+1, num_epoch, c_loss, acc) )

    # evaluate
    num_epoch = 1
    tf_iter = tf.compat.v1.data.make_initializable_iterator(valid_gen)
    tf_next = tf_iter.get_next()
    sess.run(tf_iter.initializer)
    for epoch in range(num_epoch):
        c_loss, acc = 0.0, 0.0
        for ii in range(validY.shape[0] // batch_size):
            X, Y = sess.run(tf_next)
            [b_c_loss, b_acc, _] = sess.run([model.c_loss, model.op_acc, model.train_op],
                                                feed_dict={ model.clf_input: X,
                                                            model.label_ref: Y,
                                                            tf.keras.backend.learning_phase(): 0})
            c_loss = c_loss + b_c_loss
            acc = acc + b_acc
        
        c_loss = c_loss / (validY.shape[0] // batch_size)
        acc = acc / (validY.shape[0] // batch_size)
        print('[Validation]Epoch: {:d}/{:d} - loss: {:.3f} - acc: {:.3f}'.format(epoch+1, num_epoch, c_loss, acc) )

    # evaluate
    num_epoch = 1
    tf_iter = tf.compat.v1.data.make_initializable_iterator(test_gen)
    tf_next = tf_iter.get_next()
    sess.run(tf_iter.initializer)
    for epoch in range(num_epoch):
        c_loss, acc = 0.0, 0.0
        for ii in range(testY.shape[0] // batch_size):
            X, Y = sess.run(tf_next)
            [b_c_loss, b_acc, _] = sess.run([model.c_loss, model.op_acc, model.train_op],
                                                feed_dict={ model.clf_input: X,
                                                            model.label_ref: Y,
                                                            tf.keras.backend.learning_phase(): 0})
            c_loss = c_loss + b_c_loss
            acc = acc + b_acc
        
        c_loss = c_loss / (testY.shape[0] // batch_size)
        acc = acc / (testY.shape[0] // batch_size)
        print('[Testing]Epoch: {:d}/{:d} - loss: {:.3f} - acc: {:.3f}'.format(epoch+1, num_epoch, c_loss, acc) )

self.train_op = optimizer.minimize(c_loss, var_list=[model.trainable_variables])

self.train_op = optimizer.minimize(c_loss(), var_list=[model.trainable_variables])

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

        dataset = tf.data.Dataset.from_tensor_slices( (dataX, dataY) )
        dataset = dataset.shuffle(batch_size * 8)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

class custom_model():
    def __init__(self):

        def Acc():
            acc = tf.keras.metrics.categorical_accuracy(label_ref, clf_out)
            return tf.math.reduce_mean(acc)

        def c_loss():
            loss = tf.keras.losses.categorical_crossentropy(label_ref, clf_out)
            loss = tf.math.reduce_mean(loss)
            return loss

        # create model
        clf_input = tf.keras.layers.Input(shape=(32,32,3), name="model/input")
        model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None, input_tensor=clf_input, pooling='max', classes=10)
        #model = tf.keras.applications.vgg16.VGG16(include_top=True, weights=None, input_tensor=clf_input, pooling='max', classes=10)
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

        label_ref = tf.keras.layers.Input(shape=(10,) , name='label_ref')
        clf_out = model(clf_input)

        # using tf.keras.optimizers.Nadam would get error
        #optimizer = tf.keras.optimizers.Nadam(lr=0.0005)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = optimizer.minimize(c_loss(), var_list=[model.trainable_variables])

        self.clf_model = model
        self.clf_input = clf_input
        self.label_ref = label_ref
        self.op_acc = Acc()
        self.c_loss = c_loss()

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
    #model = tf.keras.applications.vgg16.VGG16(include_top=True, weights=None, input_shape=(32,32,3), pooling=None, classes=10)
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    # fit and evalutate
    model.fit(train_gen,
            steps_per_epoch = trainY.shape[0] // batch_size,
            validation_data = valid_gen,
            validation_steps= validY.shape[0] // batch_size,
            epochs=5,
            verbose=2)
    model.evaluate(testX, testY, verbose=2, batch_size=batch_size)

    # create a new model
    print('Make sure that we create a new model.')
    model = custom_model()
    sess.run(tf.compat.v1.global_variables_initializer())
    model.clf_model.evaluate(testX, testY, verbose=2, batch_size=batch_size)

    # train model
    num_epoch = 5
    total_len = trainY.shape[0] // batch_size
    tf_iter = tf.compat.v1.data.make_initializable_iterator(train_gen)
    tf_next = tf_iter.get_next()
    sess.run(tf_iter.initializer)
    for epoch in range(num_epoch):
        c_loss, acc = 0.0, 0.0
        for ii in range(total_len):
            X, Y = sess.run(tf_next)
            [b_c_loss, b_acc, _] = sess.run([model.c_loss, model.op_acc, model.train_op],
                                                feed_dict={ model.clf_input: X,
                                                            model.label_ref: Y,
                                                            tf.keras.backend.learning_phase(): 1})
            c_loss = c_loss + b_c_loss
            acc = acc + b_acc
        
        c_loss = c_loss / total_len
        acc = acc / total_len
        print('[Training]Epoch: {:d}/{:d} - loss: {:.3f} - acc: {:.3f}'.format(epoch+1, num_epoch, c_loss, acc) )

    print('Show loss and accuracy with keras API')
    model.clf_model.evaluate(trainX, trainY, verbose=2, batch_size=batch_size)
    model.clf_model.evaluate(validX, validY, verbose=2, batch_size=batch_size)
    model.clf_model.evaluate(testX, testY, verbose=2, batch_size=batch_size)

    print('Show loss and accuracy with low level API')
    # evaluate
    num_epoch = 1
    total_len = validY.shape[0] // batch_size
    tf_iter = tf.compat.v1.data.make_initializable_iterator(valid_gen)
    tf_next = tf_iter.get_next()
    sess.run(tf_iter.initializer)
    for epoch in range(num_epoch):
        c_loss_t, acc_t, c_loss_f, acc_f = 0.0, 0.0, 0.0, 0.0
        for ii in range(total_len):
            X, Y = sess.run(tf_next)
            [b_c_loss, b_acc] = sess.run([model.c_loss, model.op_acc],
                                        feed_dict={ model.clf_input: X,
                                                    model.label_ref: Y,
                                                    tf.keras.backend.learning_phase(): 1})
            c_loss_t = c_loss_t + b_c_loss
            acc_t = acc_t + b_acc

            [b_c_loss, b_acc] = sess.run([model.c_loss, model.op_acc],
                                        feed_dict={ model.clf_input: X,
                                                    model.label_ref: Y,
                                                    tf.keras.backend.learning_phase(): 0})
            c_loss_f = c_loss_f + b_c_loss
            acc_f = acc_f + b_acc

        c_loss_t = c_loss_t / total_len
        c_loss_f = c_loss_f / total_len
        acc_t = acc_t / total_len
        acc_f = acc_f / total_len
        print('[Validation][learning_phase=1] Epoch: {:d}/{:d} - loss: {:.3f} - acc: {:.3f}'.format(epoch+1, num_epoch, c_loss_t, acc_t) )
        print('[Validation][learning_phase=0] Epoch: {:d}/{:d} - loss: {:.3f} - acc: {:.3f}'.format(epoch+1, num_epoch, c_loss_f, acc_f) )

    # evaluate
    num_epoch = 1
    total_len = testY.shape[0] // batch_size
    tf_iter = tf.compat.v1.data.make_initializable_iterator(test_gen)
    tf_next = tf_iter.get_next()
    sess.run(tf_iter.initializer)
    for epoch in range(num_epoch):
        c_loss_t, acc_t, c_loss_f, acc_f = 0.0, 0.0, 0.0, 0.0
        for ii in range(total_len):
            X, Y = sess.run(tf_next)
            [b_c_loss, b_acc] = sess.run([model.c_loss, model.op_acc],
                                        feed_dict={ model.clf_input: X,
                                                    model.label_ref: Y,
                                                    tf.keras.backend.learning_phase(): 1})
            c_loss_t = c_loss_t + b_c_loss
            acc_t = acc_t + b_acc

            [b_c_loss, b_acc] = sess.run([model.c_loss, model.op_acc],
                                        feed_dict={ model.clf_input: X,
                                                    model.label_ref: Y,
                                                    tf.keras.backend.learning_phase(): 0})
            c_loss_f = c_loss_f + b_c_loss
            acc_f = acc_f + b_acc

        c_loss_t = c_loss_t / total_len
        c_loss_f = c_loss_f / total_len
        acc_t = acc_t / total_len
        acc_f = acc_f / total_len
        print('[Testing][learning_phase=1] Epoch: {:d}/{:d} - loss: {:.3f} - acc: {:.3f}'.format(epoch+1, num_epoch, c_loss_t, acc_t) )
        print('[Testing][learning_phase=0] Epoch: {:d}/{:d} - loss: {:.3f} - acc: {:.3f}'.format(epoch+1, num_epoch, c_loss_f, acc_f) )