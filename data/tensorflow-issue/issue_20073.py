import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

FN = 'tmp.model.h5'
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(8, 8, 1)),
  keras.layers.Conv2D(1, 5),
])
keras.models.save_model(model, FN)
keras.models.load_model(FN)

model = keras.Sequential([                                                        
  keras.layers.Conv2D(1, 5, input_shape=(8, 8, 1)),                               
])

def makeModel(env, shape, fname, lr):

    # does not work due to https://github.com/tensorflow/tensorflow/issues/20073
    # if os.path.isfile(fname):
    #    print("Loading model")
    #    return load_model(fname)

    m = Sequential()
    m.add(InputLayer(input_shape=shape+(4,)))
    m.add(Conv2D(32, kernel_size=8, strides=4, activation='relu'))
    m.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
    m.add(Conv2D(64, kernel_size=3, activation='relu'))
    m.add(Dense(512, activation='relu', activity_regularizer="l2"))
    m.add(Flatten())
    m.add(Dense(env.action_space.n, activation="linear"))
    m.compile(loss=huber_loss, optimizer=tf.keras.optimizers.RMSprop(lr))
    if os.path.isfile(fname):
       m.load_weights(fname)
    return m