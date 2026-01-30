from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import time
#MODEL NAME

# print(tf.__version__)


NAME = "fashion_mnist_28x28_{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model_path="models/{}.h5".format(NAME)

# inspecting data for the fashion-mnist images 
def inspect_data():
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def pre_process_data():
    global train_images,test_images
    train_images = train_images/255.0
    test_images = test_images/255.0

def show25():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i],cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

#pre_process_data()
#show25()

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation=tf.nn.relu),
        keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

#model = create_model()
#model.summary()

def train_model():
    model = create_model()
    pre_process_data()
    model.fit(train_images,train_labels,batch_size=32,epochs=5)
    return model

def train_and_save_as_whole():
    # save the model in the HDF5
    trained_model = train_model()
    trained_model.save(model_path)

def read_model_as_whole(showsummary=False):
    model = keras.models.load_model(model_path)
    if showsummary:
        model.summary()
    return model

def read_whole_model_and_evaluate():
    model=read_model_as_whole()
    loss,acc = model.evaluate(test_images,test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


train_and_save_as_whole()