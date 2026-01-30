from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# system imports
import os
import random

# lib imports
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2
import tensorflow as tf
import sklearn
import sklearn.metrics
import tqdm
import tensorflow as tf
import os

import matplotlib.pyplot as plt

EPS = np.finfo(float).eps

#=====================================================================================================================================================
# Input parameters
#-----------------------------------------------------------------------------------------------------------------------------------------------------
batchSize = 16
imageSize = 335
tileSize = 256

#-----------------------------------------------------------------------------------------------------------------------------------------------------
def unet(numCoefs, input_size = (192,192,1), shrinkFactor = 1, name = ''):
    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(64//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv2)    
    conv3 = tf.keras.layers.Conv2D(256//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv3)    
    conv4 = tf.keras.layers.Conv2D(512//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tf.keras.layers.Dropout(0.0)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(drop4)    

    conv5 = tf.keras.layers.Conv2D(1024//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='code_'+name)(conv5)
    drop5 = tf.keras.layers.Dropout(0.0)(conv5)
    
    up6 = tf.keras.layers.Conv2D(512//shrinkFactor, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2), interpolation='bilinear')(drop5))
    merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(512//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(512//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tf.keras.layers.Conv2D(192//shrinkFactor, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2), interpolation='bilinear')(conv6))
    merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(192//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(192//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = tf.keras.layers.Conv2D(128//shrinkFactor, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2), interpolation='bilinear')(conv7))
    merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = tf.keras.layers.Conv2D(128//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(128//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = tf.keras.layers.Conv2D(64//shrinkFactor, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2), interpolation='bilinear')(conv8))
    merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = tf.keras.layers.Conv2D(64//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(64//shrinkFactor, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = tf.keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = tf.keras.layers.Conv2D(1, 1, activation = 'relu')(conv9)
    
    r = tf.keras.layers.Lambda(lambda x: (x - tf.keras.backend.min(x)) / (tf.keras.backend.max(x) - tf.keras.backend.min(x)), name = 'reconstruction_'+name)(conv10)

    model = tf.keras.models.Model(inputs, r)
    
    return model
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def buildModel():           
    numCoefs = 3
    reconstruction = unet(numCoefs, input_size=(256,256,1), shrinkFactor=4)
    model = tf.keras.models.Model(reconstruction.inputs[0], reconstruction.outputs[0])
        
    return model
#-----------------------------------------------------------------------------------------------------------------------------------------------------

#=====================================================================================================================================================
# Define and create generators
#-----------------------------------------------------------------------------------------------------------------------------------------------------
class TrainGenerator(tf.keras.utils.Sequence):
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, batchSize, tileSize, imageSize):        
        self._batchSize = batchSize
        self._imageSize = imageSize
        self._tileSize = tileSize
        self._numFiles = 1000
        return
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    def next(self):
        return self.__getitem__(0)
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):
        # idx intentionally not used
        return self._next()
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    # Returns number of steps/iterations to perform to go through all the data once
    def getStepsPerEpoch(self):
        return self._numFiles // self._batchSize 
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        return self.getStepsPerEpoch()
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    # Returns a batch of data.    
    def _next(self):         
        xRaw = np.zeros((self._batchSize, self._imageSize, self._imageSize, 1), dtype='complex64')
        # Prep for net input
        x = np.zeros((self._batchSize, self._tileSize, self._tileSize, 1), dtype='float32')
        xSlc = np.zeros((self._batchSize, self._tileSize, self._tileSize, 1), dtype='complex64')
        xDefocus = np.zeros((self._batchSize, self._tileSize, self._tileSize, 1), dtype='float32')
        y = np.zeros(self._batchSize)        
                
        for k in range(self._batchSize):
            y[k] = 1
        #              
        
        # More augmentation
        for k in range(self._batchSize):                   
            # Random Crops
            crop = self._imageSize - self._tileSize            
            startX = np.random.randint(crop)
            startY = np.random.randint(crop)
            slc = xRaw[k, startX:startX + self._tileSize, startY:startY + self._tileSize, 0]     
            x[k, :,:, 0] = 0                
        #

        return xDefocus, x
    #-------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------
class TestGenerator(tf.keras.utils.Sequence):
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, batchSize, tileSize = 256, imageSize = 335):        
        self._batchSize = batchSize
        self._tileSize = tileSize
        self._imageSize = imageSize       
        self._numFiles = 5000
        return
    #-------------------------------------------------------------------------------------------------------------------------------------------------    
    def __len__(self):
        return self.getStepsPerEpoch()    
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    # Returns number of steps/iterations to perform to go through all the data once
    def getStepsPerEpoch(self):
        return self._numFiles // self._batchSize
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):        
        return self.next(idx)
    #    
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    # Returns batch sized used to iniitalize
    def getBatchSize(self):
        return self._batchSize    
    #-------------------------------------------------------------------------------------------------------------------------------------------------
    # Returns a batch of data.
    def next(self, batch_id):
        xs = np.zeros((self._batchSize, self._tileSize , self._tileSize , 1), dtype='float32')
        xSlc = np.zeros((self._batchSize, self._tileSize , self._tileSize , 1), dtype='complex64')
        ys = np.zeros(self._batchSize)
        xsOrig = np.zeros((self._batchSize, self._tileSize , self._tileSize , 1), dtype='float32')                   

        masterOffset = (self._imageSize -self._tileSize )//2
        center = [self._imageSize //2, self._imageSize //2]
        for k in range(self._batchSize):            
            tile = np.zeros((335,335))
            
            # Center  Crops
            crop = self._imageSize  - self._tileSize             
            startX = crop//2
            startY = crop//2
            slc = tile[startX:startX + self._tileSize , startY:startY + self._tileSize]                                     

            # Label
            ys[k] = 1
            
        #
        
        return xs, xsOrig
    #-------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------
def main():
   
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    # So many random seeds
    random.seed(1)
    np.random.seed(1)
    tf.random.set_random_seed(1)
    
    #=====================================================================================================================================================
    # Create output folder of model
    #-----------------------------------------------------------------------------------------------------------------------------------------------------          
    # Create our two generators
    trainGenerator = TrainGenerator(batchSize, 256, 335)    
    testGenerator = TestGenerator(batchSize, 256, 335)
        
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    # Build model
    model = buildModel()
    model.summary()
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    # Log the model traincompileing parameters
    stepsPerEpochTrain = int(trainGenerator.getStepsPerEpoch())
    stepsPerEpochTest = testGenerator.getStepsPerEpoch()
    print('=== Fitting Model ===')
    print(' Batch size:                  {0}'.format(batchSize))
    print(' Steps per training epoch:    {0}'.format(stepsPerEpochTrain))    
    print(' Steps per testing epoch:     {0}'.format(stepsPerEpochTest))
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    
    #=====================================================================================================================================================
    # Begin training            
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=1e-3), loss = 'mse')    
    
    for epoch in range(0,9999):
        print('========== Epoch {0} =========='.format(epoch))
        
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        # Train the model
        history = model.fit_generator(trainGenerator, workers=5)         
        trainError = history.history['loss'][0]
    
        #-------------------------------------------------------------------------------------------------------------------------------------------------  
        # Dump out mosaic from a batch
        x,y = testGenerator.next(1)
               
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        # Test the model    
        r = model.evaluate_generator(trainGenerator, workers=5, verbose=1)  # idg Leaving this here for postarity. maybe keras will work someday.

        # Compute test loss
        test_err = np.mean(r)    
    # end training epoch
    return
#-----------------------------------------------------------------------------------------------------------------------------------------------------
  
if __name__== "__main__":
    main()

    print('done')