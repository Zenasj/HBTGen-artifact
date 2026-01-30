from tensorflow.keras import layers

from __future__ import print_function
import keras
from keras.datasets import cifar10,mnist,cifar100
from keras import Sequential,optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import LoggerYN as YN
import numpy as np
import scipy.io as sio
import utilsYN as uYN
import datetime
import time



def initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    global Dataset    
    global pbatchSize
    global pnumClasses
    global pEpochs
    global pLearningRate
    global pMomentum
    global pWeightDecay
    Dataset = dataset
    pbatchSize = batchSize
    pnumClasses = numClasses
    pEpochs = epochs
    pLearningRate = learningRate
    pMomentum = momentum
    pWeightDecay = weightDecay
    
def NormalizeData(x_train,x_test):
        x_train /= 255
        x_test /= 255
        return x_train, x_test

def CategorizeData(y_train,y_test,pnumClasses):
    y_train = keras.utils.to_categorical(y_train, pnumClasses)
    y_test = keras.utils.to_categorical(y_test, pnumClasses)
    return y_train, y_test
    
def loadData():

    Dataset = cifar100
    (x_train, y_train), (x_test, y_test) = Dataset.load_data(label_mode='fine') if fineFlag else Dataset.load_data()
    
    global imgRows
    global imgCols
    global inputShape
    
    imgRows = x_train.shape[1]
    imgCols = x_train.shape[2]

    try:
        imgRGB_Dimensions = x_train.shape[3]
    except Exception:
        imgRGB_Dimensions = 1 #For Gray Scale Images

    print(x_train.shape)
    x_train = x_train.reshape(x_train.shape[0], imgRows, imgCols, imgRGB_Dimensions)
    x_test = x_test.reshape(x_test.shape[0], imgRows, imgCols, imgRGB_Dimensions)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = NormalizeData(x_train, x_test)
    y_train, y_test = CategorizeData(y_train,y_test,pnumClasses)
    inputShape = (imgRows, imgCols, imgRGB_Dimensions)
    return x_train, y_train, x_test, y_test


def model_CIFAR100():
    CIFAR_model = Sequential()
    CIFAR_model.add(Conv2D(128, (3, 3), padding='same',strides=1,input_shape=inputShape))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(Conv2D(128, (3, 3)))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='valid'))
    CIFAR_model.add(Dropout(0.1))
    
    CIFAR_model.add(Conv2D(256, (3, 3), padding='same',strides=1))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(Conv2D(256, (3, 3)))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='valid'))
    CIFAR_model.add(Dropout(0.25))
    
    CIFAR_model.add(Conv2D(512, (3, 3), padding='same',strides=1))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(Conv2D(512, (3, 3)))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='valid'))
    CIFAR_model.add(Dropout(0.5))
    
    
    CIFAR_model.add(Flatten())
    CIFAR_model.add(Dense(1024))
    CIFAR_model.add(Activation('relu'))
    CIFAR_model.add(Dropout(0.5))
    CIFAR_model.add(Dense(pnumClasses))
    CIFAR_model.add(Activation('softmax'))    
    return CIFAR_model


def evaluateModel(model,x_test,y_test,verbose):
    pLoss, pAcc = model.evaluate(x_test, y_test, verbose)
    print("Test Loss", pLoss)
    print("Test Accuracy", pAcc)
     


def RunCIFAR100(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay):
    initParameters(dataset,batchSize,numClasses,epochs,learningRate,momentum,weightDecay)
    x_train, y_train, x_test, y_test = loadData()
    CIFAR_model = model_CIFAR100()
    CIFAR_sgd = optimizers.SGD(lr=learningRate, decay=weightDecay, momentum=momentum, nesterov=False)
    CIFAR_model.compile(loss='categorical_crossentropy',optimizer=CIFAR_sgd, metrics=['accuracy'])
    CIFAR_model.fit(x_train, y_train,batch_size=batchSize,epochs=epochs,validation_data=(x_test, y_test),shuffle=True)
    evaluateModel(CIFAR_model,x_test, y_test, verbose=1)


def runModel(dataset,batchSize=128,numClasses=10,epochs=12,learningRate=0.01,momentum=0.5,weightDecay=1e-6):
    RunCIFAR100(dataset,batchSize,numClasses=100,epochs=epochs,learningRate=learningRate,momentum=momentum,weightDecay=weightDecay)

def main():
    runModel("cifar100",epochs=200)