import numpy as np
import math
from tensorflow import keras
from tensorflow.keras import optimizers

class SoftDiceLoss(tf.keras.losses.Loss):
    '''
    SoftDiceLoss calculates multi-class soft dice loss
    loss = avg_batch(1-(sum(W_k*sum(yPred.*yTrue)))/(sum(W_ksum(yPred^2+yTrue^2))))
    where W_k = 1/(number of voxels in class k)^wPow
    Class number of segmented regions includes background
    Args:
        yPred/yTrue: prediced and desired outputs shaped as [mbSize, classNum, tensor dimensions]. Also, both must be float-point
    	wPow, power of weiight. A higher one favours classes with a smaller number of voxels
    Return:
        loss: a scalar tensor
    '''
    def __init__(self, wPow=2.0, name='SoftDiceLoss'):
        super().__init__(name=name)
        self.epsilon = 1e-16 
        self.wPow = wPow

    def call(self, yPred, yTrue):
        yTrue =tf.dtypes.cast(yTrue, dtype=yPred.dtype)
		# Dot product yPred and yTrue and sum them up for each datum and class
        crossProd=tf.multiply(yPred, yTrue)
		# As a symbolic tensor, dimensions and shapes etc. cannot be extracted from data, nor can it be used in subroutines.
        crossProdSum=tf.math.reduce_sum(crossProd, axis=np.arange(2, 5)) #tf.rank(yTrue)))
		# Calculate weight for each datum and class 
        weight = tf.math.reduce_sum(yTrue, axis=np.arange(2, 5))#tf.rank(yTrue)))
		#weight = tf.math.divide(1, tf.math.square(weight)+self.epsilon)
        weight = tf.math.divide(1, tf.math.pow(weight, self.wPow)+self.epsilon)
		# Weighted sum over classes
        numerator = 2*tf.math.reduce_sum(tf.multiply(crossProdSum, weight), axis=1)
		# Saquared summation 
        yySum = tf.math.reduce_sum(tf.math.square(yPred) + tf.math.square(yTrue), axis=np.arange(2, 5))#tf.rank(yTrue)))
		# Weighted sum over classes
        denominator = tf.math.reduce_sum(tf.multiply(weight, yySum), axis=1)
		# Get individual loss and average over minibatch
        loss = tf.math.reduce_mean(1 - tf.math.divide(numerator, denominator+self.epsilon))
			
        return loss
    
    def get_config(self):
        config = super(SoftDiceLoss, self).get_config()
        return config

curOpt = tf.keras.optimizers.Adam(learning_rate=1e-4)	
lossFunc=SoftDiceLoss(2.0)
ckpt = tf.train.Checkpoint(model=myModel(...), optimizer=curOpt, lossFunc=lossFunc, accFunc=accFunc)

import tensorflow as tf
from tensorflow.keras import layers, activations

class C3BR(tf.keras.Model):
    def __init__(self, filterNum, kSize, strSize, padMode, dFormat='channels_first'):
        super(C3BR, self).__init__()
        if dFormat == 'channels_first':
            self.conAx = 1
        else:
            self.conAx = -1
        self.kSize = (kSize, kSize, kSize)
        self.conv = layers.Conv3D(filters=filterNum, kernel_size=self.kSize, strides=strSize, padding=padMode, data_format=dFormat)
        self.BN = layers.BatchNormalization(axis=self.conAx)
        self.Relu = layers.ReLU()
    
    def call(self, inputs, ifTrain=False):
        x = self.conv(inputs)
        x= self.BN(x, training=ifTrain)
        outputs = self.Relu(x)
        return outputs

    def build_model(self, input_shape):
        ''' A work-around to define dimensions of signals through the NN'''
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(inputs, True) 


class lossExample(tf.keras.losses.Loss):

    def __init__(self, name='lossExample'):
        super().__init__(name=name)
		
    def call(self, yPred, yTrue):
			
        return tf.reduce_mean(yPred - yTrue)
    

curOpt = tf.keras.optimizers.Adam(learning_rate=1e-4)	
lossFunc=lossExample()
ckpt = tf.train.Checkpoint(model=C3BR(32, 3, 1, 'valid'), optimizer=curOpt, lossFunc=lossFunc)

@tf.function
def lossExample(yPred, yTrue):
    return tf.reduce_mean(yPred-yTrue)

curOpt = tf.keras.optimizers.Adam(learning_rate=1e-4)	
ckpt = tf.train.Checkpoint(model=C3BR(32, 3, 1, 'valid'), optimizer=curOpt, lossFunc=lossExample)