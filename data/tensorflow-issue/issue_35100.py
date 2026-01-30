import math
import tensorflow as tf
from tensorflow import keras

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, dir, n_classes):
        'Initialization'
        config = configparser.ConfigParser()
        config.sections()
        config.read('config.ini')

        self.dim = (int(config['Basics']['PicHeight']),int(config['Basics']['PicWidth']))
        self.batch_size = int(config['HyperParameter']['batchsize'])
        self.labels = labels
        self.list_IDs = list_IDs
        self.dir = dir
        self.n_channels = 3
        self.n_classes = n_classes
        self.on_epoch_end()        


    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.floor(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y, [None]