from tensorflow import keras
from tensorflow.keras import models

import os
import random
import numpy as np
from PIL import Image
import multiprocessing 
import tensorflow as tf
from resizeimage import resizeimage

def images_list(path):
    images_list = []
    XY = []
    with open(path,"r") as file:
        images_list = file.read().split('\n')
        XY = [row.split(" ") for row in images_list if len(row.split(" ")) > 1]
    return np.asarray(XY)

def load_images(X,Y,i):
    root = 'E:\\images\\rvl-cdip\\rvl-cdip\\images'
    img_matrixes = []
    labels = []
    length = len(X)
    for index in range(len(X)):
        matrix = Image.open(os.path.join(root, X[index].replace('/',"\\")))
        img_matrixes.append(resize_image_500(matrix))
        labels.append(Y[index])
        
    img_matrixes = np.asarray(img_matrixes)
    labels = np.asarray(labels)
    
    assert len(img_matrixes) == len(labels)
          
    #print("{}: Loaded {} images".format(i,length))
          
    return np.reshape(img_matrixes,(img_matrixes.shape[0],500,500,1)),labels

def resize_image(img):
    np_img = np.asarray(img)
    
    if(np_img.shape[1] < 3235):
        missing_width = 3235 - np_img.shape[1]
        white_matrix = np.empty((1000,missing_width),dtype=float)
        white_matrix.fill(255)
        np_img = np.hstack((np_img, white_matrix))
        
    assert np_img.shape[0] == 1000
    assert np_img.shape[1] == 3235
    
    return np_img

def resize_image_500(img):
    resized = resizeimage.resize_cover(img, [500, 500])
    np_img = np.asarray(resized)
    
    assert np_img.shape[0] == 500
    assert np_img.shape[1] == 500
    
    return np_img

def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    i = 0 
    for start_idx in np.arange(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        i+=1
        yield load_images(inputs[excerpt], targets[excerpt],i)

from tensorflow.keras import layers

model = tf.keras.models.Sequential()
#H1
model.add(layers.Conv2D(8, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#H2
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#H3
model.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#H4
model.add(layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#H5
model.add(layers.Flatten())
#Dense
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer='adam',
              metrics=[tf.keras.metrics.Accuracy()])
#Training data
XY_train =  images_list(train_path)
X_train = XY_train[:,0]
Y_train = XY_train[:,1].astype(int)

#Testing data
XY_test =  images_list(test_path)
X_test = XY_test[:,0]
Y_test = XY_test[:,1].astype(int)

#Validation data
XY_val = images_list(valid_path)
X_val = XY_val[:,0]
Y_val = XY_val[:,1].astype(int)

batch_size = 750
history = model.fit_generator(generator=iterate_minibatches(X_train, Y_train,batch_size),
                                  validation_data=iterate_minibatches(X_test, Y_test, batch_size),
                                  # validation_data=None,
                                  steps_per_epoch=len(X_train)//batch_size,
                                  validation_steps=len(X_test)//batch_size,
                                  verbose=1,
                                  epochs=100,
                                  use_multiprocessing=True,
                                  workers=multiprocessing.cpu_count() 
                             )