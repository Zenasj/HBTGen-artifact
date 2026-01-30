from tensorflow.keras import models

import os
import csv
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

# Get list of physical GPU devices
gpu_list = tf.config.list_physical_devices('GPU')

if len(gpu_list) > 0:
    # Check Number of GPUs
    print('number of GPUs available:', len(gpu_list))
    print('\nGPU name:')

    # Check GPU Name
    for i in range(len(gpu_list)):
        print(str(i + 1) + '.', gpu_list[i].name.split(':', 1)[1])

    # Set memory growth for the GPU
    tf.config.experimental.set_memory_growth(gpu_list[0], True)

    # Set visible devices to only use the first GPU
    tf.config.experimental.set_visible_devices(gpu_list[0], 'GPU')

    # Verify that the GPU is being used
    print('\nUsing GPU:', gpu_list[0])

# Set attribute variable
attr = 'gender'

# Set directories
root_dir = '/mnt/c/Users/Ang/Desktop/11 DL/Assignment/DeepFashion/images'
data_dir = os.path.join(root_dir, 'data')
label_path = os.path.join(data_dir, attr + ' label.csv')
model_path = os.path.join(data_dir, attr + ' model.h5')

# Set data augmentation for train set
train_generator = ImageDataGenerator(
    rescale = 1./255,         # Normalize the data
    rotation_range = 0,       # Randomly rotate images by up to certain degrees
    width_shift_range = 0,    # Randomly shift images horizontally by up to certain percentage of the width
    height_shift_range = 0,   # Randomly shift images vertically by up to certain percentage of the height
    shear_range = 0,          # Randomly apply shear transformation with a max shear of certain percentage
    zoom_range = 0,           # Randomly zoom in/out of images by up to certain percentage
    horizontal_flip = True,   # Randomly flip images horizontally
    vertical_flip = False,    # Do not randomly flip images vertically
    fill_mode = 'nearest'     # Fill any newly created pixels with the nearest pixel value
)

# Data augmentation not applicable to validate and test set
validate_test_generator = ImageDataGenerator(rescale = 1./255)

# Set directories
train_dir = os.path.join(data_dir, 'train', attr)
validate_dir = os.path.join(data_dir, 'validate', attr)
test_dir = os.path.join(data_dir, 'test', attr)

# Set variables
target_size = (110, 75)
batch_size = 10

# Import and generate the image data for train set
train_set = train_generator.flow_from_directory(
    train_dir, 
    target_size = target_size, 
    color_mode = 'rgb', 
    class_mode = 'categorical',
    batch_size = batch_size,
    shuffle = True, 
    seed = 0
)

# Import and generate the image data for validate set
validate_set = validate_test_generator.flow_from_directory(
    validate_dir, 
    target_size = target_size, 
    color_mode = 'rgb', 
    class_mode = 'categorical', 
    batch_size = batch_size, 
    shuffle = True, 
    seed = 0
)

# Import and generate the image data for test set
test_set = validate_test_generator.flow_from_directory(
    test_dir, 
    target_size = target_size, 
    color_mode = 'rgb', 
    class_mode = 'categorical', 
    batch_size = batch_size, 
    shuffle = True, 
    seed = 0
)

# Save the label code to csv file
label_dict = train_set.class_indices

with open(label_path, 'w', newline = '') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['code', 'label'])

    for label, code in label_dict.items():
        writer.writerow([code, label])
        
# Get the classes for all sets
train_classes = train_set.num_classes
validate_classes = validate_set.num_classes
test_classes = test_set.num_classes

# Get the shape for all sets
train_shape = train_set.image_shape
validate_shape = validate_set.image_shape
test_shape = test_set.image_shape

# Print the classes and shape for all sets
print()
print(label_dict)
print()
print('train classes:', train_classes)
print('validate classes:', validate_classes)  
print('test classes:', test_classes)
print()
print('train shape:', train_shape)
print('validate shape:', validate_shape)
print('test shape:', test_shape)

# Set variables
c1 = 16
c2 = 32
h1 = 64
activation = 'relu'

# Build model
model = Sequential()
model.add(Conv2D(c1, (3, 3), input_shape = train_shape, 
                 padding = 'same', activation = activation))
model.add(Conv2D(c2, (3, 3), input_shape = train_shape, 
                 padding = 'same', activation = activation))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(h1, activation = activation))
model.add(Dense(train_classes, activation = 'softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Show the model summary
model.summary(line_length = 80)

# Train the model
print('\nModel Training:')
history = model.fit(train_set, validation_data = validate_set, epochs = 10, verbose = 2)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers