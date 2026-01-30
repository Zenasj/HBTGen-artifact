import numpy as np
import random

def on_epoch_end(self):
        'Updates indices after each epoch'

        #My parameter should be updated here

        print("counted", self.counter, self.genMode)
        self.counter += 1

        self.indexes = np.arange(len(self.imageID))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, augParams, genMode, batchSize=32, dim=(360,640), shuffle=True):
          self.genMode = genMode
          self.counter = 1
  

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.imageID) / self.batchSize))

    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batchSize:(index+1)*self.batchSize]

        # Find list of IDs
        imageIDtemp = [self.imageID[k] for k in indexes]
        annoIDtemp = [self.annoID[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(imageIDtemp, annoIDtemp)
        return X, Y
    
    def on_epoch_end(self):
        'Updates indices after each epoch'
        #My parameter should be updated here

        print("counted", self.counter, self.genMode)
        self.counter += 1

        self.indexes = np.arange(len(self.imageID))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

def on_epoch_end(self):
        'Updates indices after each epoch'
        traceback.print_stack()
        #My parameter should be updated here

        print("counted", self.counter, self.genMode)
        self.counter += 1

        self.indexes = np.arange(len(self.imageID))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)