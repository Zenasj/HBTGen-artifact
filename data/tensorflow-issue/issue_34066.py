import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import numpy as np

batch_size = 32

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

NUM_WORKERS = strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = batch_size * NUM_WORKERS
N_CAT = 47

def some_func(*args: tf.Tensor):
    tensor_dict_x, tensor_dict_y = {}, {}

    for index in range(1):
        tensor_dict_x[
            f"input_{index+1}"
        ] = tf.expand_dims(args[index], axis=-1)
        tensor_dict_y[
            f"dense"
        ] = tf.expand_dims(args[index], axis=-1)

    return tensor_dict_x, tensor_dict_y

def read_data():
    train_data = np.random.randint(1,N_CAT, size=9000)
    val_data = np.random.randint(1,N_CAT, size=999)

    dataset_train = (tf.data.Dataset.from_tensor_slices(train_data)
                     .prefetch(-1)
                     .map(map_func=some_func)
                     .batch(batch_size=GLOBAL_BATCH_SIZE)
                     .shuffle(1000)
                     .repeat())
    dataset_val = (tf.data.Dataset.from_tensor_slices(val_data)
                   .prefetch(-1)
                   .map(map_func=some_func)
                   .batch(batch_size=GLOBAL_BATCH_SIZE)
                   .shuffle(1000)
                   .repeat())

    return dataset_train, dataset_val

with strategy.scope():
    optimizer = Adam(lr=0.1)
    loss = losses.sparse_categorical_crossentropy
    model = build_and_compile_model(optimizer, loss)

dataset_train, dataset_val = read_data()
model.fit(x=dataset_train,
          epochs=5,
          steps_per_epoch=9000//batch_size,
          validation_data=dataset_val,
          validation_steps=999//batch_size,
)

def build_and_compile_model(optimizer, loss):
    my_input = tf.keras.layers.Input(shape=(1,))
    my_dense = tf.keras.layers.Dense(N_CAT)(my_input)

    model = tf.keras.Model(my_input, my_dense)

    model.compile(optimizer=optimizer,loss=loss)

    return model

def build_and_compile_model(optimizer, loss):

    my_input = tf.keras.layers.Input(shape=(1,))
    emb_layer = tf.keras.layers.Embedding(N_CAT,5)
    emb_inp = emb_layer(my_input)
    my_dense = tf.keras.layers.Dense(N_CAT)(emb_inp)

    model = tf.keras.Model(my_input, my_dense)

    model.compile(optimizer=optimizer,loss=loss)

    return model