import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

import numpy as np

(train_images, train_labels), _ = mnist.load_data()

num_images, img_x, img_y = train_images.shape

# linearize images
train_images = train_images.reshape( (num_images, img_x * img_y) )
train_images = train_images.astype("float64") / 255.0
# one-hot-encoding of labels
train_labels = to_categorical(train_labels)


# initialize the network
model = Sequential()

# add nodes to the network
model.add( Dense(15, input_shape=(img_x*img_y,) # input size
                ) )
model.add( Activation("sigmoid") )

model.add( Dense(10) )
model.add( Activation("softmax") )

# finalize the network
model.compile( optimizer="rmsprop",
               loss='categorical_crossentropy',
               metrics=['acc'] )

# train the network
hist = model.fit( x=train_images, # training examples
                  y=train_labels, # desired output
                  epochs=10,      # number of training epochs 
                  verbose=1)

with tf.device("/gpu:0"):
    matrix = np.ones((1024, 1024))
    tf.linalg.matmul(matrix, matrix)