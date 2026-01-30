import math
import random
from tensorflow import keras
from tensorflow.keras import optimizers

path = r'...\TUNet_Test\Weight_03-01-46-02-01-2020_Ep00000'
model.load_weights(path)

from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import os
from time import gmtime, strftime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations
import matplotlib.pyplot as plt


class C3BR(tf.keras.Model):
    ''' 3D Convolution + Batch Normalisation + Relu '''
    def __init__(self, filterNum, kSize, strSize, padMode):
        super(C3BR, self).__init__()
        self.conv = layers.Conv3D(filters=filterNum, kernel_size=kSize, strides=strSize, padding=padMode, data_format='channels_first')
        self.BN = layers.BatchNormalization(axis=1)
    
    def call(self, inputs, ifTrain=False):
        x = self.conv(inputs)
        if ifTrain == True:
            x = self.BN(x)
        return activations.relu(x)

    def build_model(self, input_shape):
        ''' A work-around to define dimensions of signals through the NN'''
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(inputs) 


class SimpleUNet(tf.keras.Model):
    """
    Serialise basic units so as to build up a double-layered encoder-decoder U-Net. There are many forms of U-Net. Sometimes 
    people replace Max Pooling with stride=2; and only 1 C3BR is contained in each encoder/decoder, and only 3DTransposeConv
    is used in decoder. 
    NVIDIA Tensor Cores in GPU's require certain dimensions of tensors to be a multiple of 8, so it is a best practice to choose
    units in dense layers, numbers of feature mapes in convolutional layers, as well as sizes of minibatches to comply with the
    requirement.Regardless of what model ends in, make sure the output is float32
    Input:
        inDim: (for initialisation) [modaility/channel, tensor dimensions]
        classNum: background included
        name: name for the net
        inputs: 5D tf tensor of [mbSize, modaility/channel, tensor dimensions]. Inputs must be organised into channel first order
        input_shape: a 1X5 tuple (mbSize, modaility/channel, tensor dimensions)
        ifTrain: True for training, and False for validation and testing
    Returns:
        outputs: 5D tf tensor of [mbSize, classNum, tensor dimensions]
    """
    def __init__(self, inDim, classNum):
        super(SimpleUNet, self).__init__()
        self.inDim = inDim
        self.classNum = classNum
        dimEnSt1End = np.array(inDim)[1:]-2-2
        dimEnSt2Ed = dimEnSt1End/2-2-2
        dimBridgeEnd = (dimEnSt2Ed/2-2-2)*2
        dimDEStd1End = (dimBridgeEnd-2-2)*2
        self.outDim = dimDEStd1End-2-2-2
        temp = ((dimEnSt2Ed - dimBridgeEnd)/2).astype('int32')
        crop3d1 = tuple(np.tile(temp, (2, 1)).T)
        temp = ((dimEnSt1End - dimDEStd1End)/2).astype('int32')
        crop3d2 = tuple(np.tile(temp, (2, 1)).T)

        self.en_st1_cbr1 = C3BR(32, 3, 1, 'valid')
        self.en_st1_cbr2 = C3BR(64, 3, 1, 'valid')
        self.en_st2_mp = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_first')
        self.en_st2_cbr1 = C3BR(128, 3, 1, 'valid')
        self.en_st2_cbr2 = C3BR(128, 3, 1, 'valid')
        self.bridge_mp = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_first')
        self.bridge_cbr1 = C3BR(256, 3, 1, 'valid')
        self.bridge_cbr2 = C3BR(256, 3, 1, 'valid')    
        self.bridge_tconv1 = layers.Conv3DTranspose(512, 2, strides=2, padding='valid', data_format='channels_first')
        self.de_3dcrop1 = layers.Cropping3D(crop3d1, data_format='channels_first')
        self.de_st1_concat = layers.Concatenate(axis=1)
        self.de_st1_cbr1 = C3BR(256, 3, 1, 'valid')
        self.de_st1_cbr2 = C3BR(128, 3, 1, 'valid')    
        self.de_st1_tconv1 = layers.Conv3DTranspose(128, 2, strides=2, padding='valid', data_format='channels_first')
        self.de_3dcrop2 = layers.Cropping3D(crop3d2, data_format='channels_first')
        self.de_st2_concat = layers.Concatenate(axis=1)
        self.de_st2_cbr1 = C3BR(64, 3, 1, 'valid')
        self.de_st2_cbr2 = C3BR(64, 3, 1, 'valid') 
        self.final_conv3D = layers.Conv3D(filters=self.classNum, kernel_size=3, strides=1, padding='valid', data_format='channels_first')                
    
    # @tf.function
    # In fact, decorating it does not bring much benefit as it primarily contains large ops. that have been optimised by tf.
    def call(self, inputs, ifTrain=False):
        x0 = self.en_st1_cbr1(inputs, ifTrain)
        xEnSt1End = self.en_st1_cbr2(x0, ifTrain)
        x1 = self.en_st2_mp(xEnSt1End)
        x2 = self.en_st2_cbr1(x1, ifTrain)
        xEnSt2Ed = self.en_st2_cbr2(x2, ifTrain)
        x3 = self.bridge_mp(xEnSt2Ed)  
        x4 = self.bridge_cbr1(x3, ifTrain)
        x5 = self.bridge_cbr2(x4, ifTrain)    
        xBridgeEnd = self.bridge_tconv1(x5)
        xCrop1 = self.de_3dcrop1(xEnSt2Ed)
        x6 = self.de_st1_concat([xBridgeEnd, xCrop1])
        x7 = self.de_st1_cbr1(x6, ifTrain)
        x8 = self.de_st1_cbr2(x7, ifTrain)
        xDeSt1End = self.de_st1_tconv1(x8)
        xCrop2 = self.de_3dcrop2(xEnSt1End)
        x9 = self.de_st2_concat([xDeSt1End, xCrop2])
        x10 = self.de_st2_cbr1(x9, ifTrain)
        x11 = self.de_st2_cbr2(x10, ifTrain)
        x12 = self.final_conv3D(x11)
        outputs = tf.dtypes.cast(activations.softmax(x12, axis=1), dtype=tf.float32)
        
        return outputs
        
    def build_model(self, input_shape):
        ''' A work-around to permit one to see dimensions of signals through the NN. An imperative API does not support it
            by its own right
        '''
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(inputs)
        
    def compute_output_shape(self):
        ''''Override this function if one expects to use the subclassed model in Kera's fit() method; Otherwise, it is optional.
        '''
        return tf.TensorShape(np.append(self.classNum, self.outDim))    



@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.uint8)])
def softDiceLoss(yPred, yTrue):
    '''
    SoftDiceLoss calculates multi-class soft dice loss
    loss = avg_batch(1-(sum(W_k*sum(yPred.*yTrue)))/(sum(W_ksum(yPred^2+yTrue^2))))
    where W_k = 1/(number of voxels in class k)^2
    Class number of segmented regions includes background
    Input:
        yPred/yTrue: prediced and desired outputs shaped as [mbSize, classNum, tensor dimensions]. Also, both must be float-point
    Return:
        loss: a scalar tensor
    '''
    epsilon = 1e-16 
    yTrue =tf.dtypes.cast(yTrue, dtype=yPred.dtype)
    # Dot product yPred and yTrue and sum them up for each datum and class
    crossProd=tf.multiply(yPred, yTrue)
    # As a symbolic tensor, dimensions and shapes etc. cannot be extracted from data, nor can it be used in subroutines.
    crossProdSum=tf.math.reduce_sum(crossProd, axis=np.arange(2, 5)) #tf.rank(yTrue)))
    # Calculate weight for each datum and class 
    weight = tf.math.reduce_sum(yTrue, axis=np.arange(2, 5))#tf.rank(yTrue)))
    weight = tf.math.divide(1, tf.math.square(weight)+epsilon)
    # Weighted sum over classes
    numerator = 2*tf.math.reduce_sum(tf.multiply(crossProdSum, weight), axis=1)
    # Saquared summation 
    yySum = tf.math.reduce_sum(tf.math.square(yPred) + tf.math.square(yTrue), axis=np.arange(2, 5))#tf.rank(yTrue)))
    # Weighted sum over classes
    denominator = tf.math.reduce_sum(tf.multiply(weight, yySum), axis=1)
    # Get individual loss and average over minibatch
    loss = tf.math.reduce_mean(1 - tf.math.divide(numerator, denominator+epsilon))
    
    return loss

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.uint8),))
def calcAcc(yPred, yTrue):
    '''
    Calculate accuracy class by class. For each class, only an identical label on the voxel belonging to the corresponding class 
    (1 in yTrue[class] and 0 in ~yTrue[class]) is considered being correctly identified, that is,
    Acc = (TP +TN) / (TP + FP + TN + FN), where 
    TP = sum(yPred.*YTrue); whilst TN = sum((1-yPred).*(1-yTrue))
    Input:
        yPred/yTrue: prediced and desired outputs shaped as [mbSize, classNum, tensor dimensions]. Also, both must be float-point
    Return:
        acc: a 1XclassNum tensor indicating the accuracy over each class (background included). For dichotmous segmentation, 
        the two resulting accuracies are equal
    '''
    yTrue = tf.dtypes.cast(yTrue, dtype=yPred.dtype)
    yPredInt = tf.round(yPred)
    acc = 2 * tf.math.reduce_sum(tf.math.multiply(yPredInt, yTrue), axis=np.arange(2, tf.rank(yTrue)))
    acc -= tf.math.reduce_sum(yPredInt+yTrue, axis=np.arange(2, tf.rank(yTrue)))
    acc = acc/tf.dtypes.cast(tf.math.reduce_prod(tf.shape(yTrue)[2:]), dtype=yPred.dtype)
    acc += 1
    
    return tf.math.reduce_mean(acc, axis=0)

## For debugging tf.functions
tf.config.experimental_run_functions_eagerly(True)
## Hyper-paras for data
# number of segmented classes including background
classNum = 2
modalNum = 4
imgPatchSize = [64, 64, 64]
labelPatchSize=[22, 22, 22]

## Hyper-paras for model and training
modelInDim = (modalNum,) + tuple(imgPatchSize)
epoch = 10
mbSize = 1
lr = 1e-4
lossFunc = softDiceLoss
accFunc = calcAcc
curOpt = tf.keras.optimizers.Adam(learning_rate=lr)
saveOpt =[True, 10, 'w', r'.\TUNet_Test', ]

TUNet = SimpleUNet(modelInDim, classNum)
TUNet.build_model(input_shape=(mbSize,)+modelInDim)
TUNet.summary()

@tf.function
def trainOneSample(data, model, optimizer, lossFunc, accFunc):
    ''' Evaluate model to produce generalised error/accuracy
    Input:
        data: one pair of serialised input and label from dataset
        model: an NN model for the inference       
        optimizer: an optimiser configuration used for the model
        lossFunc/accFunc: names of loss and accuracy functions
    Return:
        totalLoss/total/Acc: two vectors of loss and accuracy of each test input 
    '''
    with tf.GradientTape() as tape:
        yPred = model(data[0], ifTrain=True)
        # Calculate loss and accuracy
        curLoss = lossFunc(yPred, data[1])
        curAcc = accFunc(yPred, data[1])
    gradients = tape.gradient(curLoss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return curLoss, curAcc

for ep in np.arange(epoch):

	# Each step crosses one minibatch
	for stepTr, data in enumerate(dsTrain):
		curLoss, curAcc = trainOneSample(data, TUNet, curPot, softDiceLoss, calcAcc)

	if saveOpt[0] == True and ep % saveOpt[1] == 0:
		curTime = strftime("%H-%M-%S-%d-%m-%Y", gmtime())
		if saveOpt[2] == 'M' or saveOpt[2] == 'm':
			print('Save model at ' + curTime)
			if ep == 0:
				# Warm-up for saving model
				x = tf.random.uniform((mbSize, modalNum, 64, 64, 64), dtype=tf.float32)
				_ = TUNet(x, ifTrain=True)
				TUNet._set_inputs(x)    
			fullPath =  os.path.join(saveOpt[3], 'Model_' + curTime + '_Ep' + str(ep).zfill(5))
			os.makedirs(fullPath)
			model.save(fullPath, save_format='tf') 
		else:
			print('Save weights at ' + curTime)
			# Note the last string is name not folder
			fullPath =  os.path.join(saveOpt[3], 'Weight_' + curTime + '_Ep' + str(ep).zfill(5))
			os.makedirs(fullPath)
			model.save_weights(os.path.join(fullPath, 'Weight'), save_format='tf')

x = tf.random.uniform((mbSize, modalNum, 64, 64, 64), dtype=tf.float32)
_ = TUNet(x, ifTrain=True)
TUNet._set_inputs(x)