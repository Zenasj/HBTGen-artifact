import math
import random
from tensorflow import keras
from tensorflow.keras import layers

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

for i in range(0,self._Epochs):
            print("Epoch {} of {}".format(i+1,self._Epochs))
            self.model.fit(x=training_generator,
                        use_multiprocessing=False,
                        workers=6, 
                        epochs=1, 
                        steps_per_epoch = len(training_generator),
                        callbacks=[LoggingCallback(self.logger.debug)])

import tensorflow as tf
import datagenerator

PicX = 300
PicY = 300
Color = (255,255,255)

def main():
    print("Starting a minimal, self-contained error reproduction")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(PicX, PicY, 3)))
    model.add(tf.keras.layers.Dense(600, activation='relu'))    
    model.add(tf.keras.layers.Dense(150, activation='relu'))        
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    training_generator = datagenerator.DataGenerator(100, PicX, PicY, Color)
    print("Starting training")
    model.fit(x=training_generator, workers=1, epochs=50, steps_per_epoch = len(training_generator))
    print("Fit without error with one worker!")
    model.fit(x=training_generator, workers=6, epochs=50, steps_per_epoch = len(training_generator))
    print("Fit without error with six worker!") #For me it crashed before

if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, BatchSize, PicX, PicY, Color):
        self._BatchSize = BatchSize
        self._dim = (PicX, PicY)
        self._Color = Color
        
    def __len__(self):
        return 100
        
    def create_random_form(self):
        img = Image.new('RGB', self._dim, (50,50,50))
        draw = ImageDraw.Draw(img)
        label = np.random.randint(3)
        x0 = np.random.randint(int((self._dim[0]-5)/2))+1
        x1 = np.random.randint(int((self._dim[0]-5)/2))+int(self._dim[0]/2)
        y0 = np.random.randint(int((self._dim[1]-5)/2))
        y1 = np.random.randint(int((self._dim[1]-5)/2))+int(self._dim[1]/2)
        if label == 0:
            draw.rectangle((x0,y0,x1,y1), fill=self._Color)
        elif label == 1:
            draw.ellipse((x0,y0,x1,y1), fill=self._Color)                
        else:
            draw.polygon([(x0,y0),(x0,y1),(x1,y1)], fill=self._Color)     
        return img, label
        
    def __getitem__(self, index):
        X = np.empty((self._BatchSize, *self._dim, 3))
        y = np.empty((self._BatchSize), dtype=int)
        for i in range(0,self._BatchSize):
            img, label = self.create_random_form()
            X[i,] = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            y[i] = label
        return X, y