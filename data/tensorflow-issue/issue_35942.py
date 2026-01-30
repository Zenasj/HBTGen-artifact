import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

python 
import tensorflow as tf 
import os
import contextlib
import numpy as np
import tensorflow.keras as keras  

def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy')
    return model

def get_model_path():
    model_dir = '/tmp/m' + str(np.random.randint(0, 1000000))
    os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model')
    return model_path + ".h5"

def attempt_save_and_reload(model_path, distributed_training=False):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    with strategy.scope() if distributed_training else contextlib.nullcontext():
        model = get_model()
        model.fit(
            train_images,
            train_labels,
            epochs=1,
        )
        model.save(model_path)
        model = tf.keras.models.load_model(model_path)

if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy()
    for distributed_training in [False, True]:
        print('distributed training: ', distributed_training)
        model_path = get_model_path()
        try:
            attempt_save_and_reload(model_path, distributed_training)
        except Exception as e:
            print('Exception raised: \n', e)
        print()

new_model = tf.keras.models.load_model('model.h5')
new_model.summary()

import efficientnet.tfkeras
from tensorflow.keras.models import load_model

model = load_model('path/to/model.h5')