from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import time as tm

INPUT_SHAPE=[3, 5]
NUM_POINTS=20
BATCH_SIZE=7
EPOCHS=4

def data_gen(num, in_shape):
    for i in range(num):
        x = np.random.rand(in_shape[0], in_shape[1])
        y = random.randint(0,2)
        yield x, y
        
def data_gen_all(num, in_shape, num_labels):
    x = np.zeros([num]+in_shape)
    y = np.zeros([num]+[num_labels])
    for i in range(num):
        x[i,:,:]= np.random.rand(in_shape[0], in_shape[1])
        y[i]= tf.one_hot(random.randint(0, num_labels), num_labels).numpy()
    return x, y

train = tf.data.Dataset.from_generator(
    generator=data_gen,
    output_types=(tf.float32, tf.int32),
#     output_shapes=(tf.TensorShape([None, INPUT_SHAPE[1]]), tf.TensorShape(None)),
#     output_shapes=(tf.TensorShape(INPUT_SHAPE), tf.TensorShape(())),
    output_shapes=([None, INPUT_SHAPE[1]],()),
    args=([NUM_POINTS, INPUT_SHAPE])
)

def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation="tanh",input_shape=input_shape),        
        tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer= tf.keras.regularizers.l2(0.001))
    ])
    return model

model = create_model(input_shape=INPUT_SHAPE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0),
    loss= tf.keras.losses.SparseCategoricalCrossentropy(),
#     loss= tf.keras.losses.CategoricalCrossentropy()
    )
print(model.summary())
model.fit(train.batch(BATCH_SIZE), epochs=EPOCHS, verbose=2)
model.evaluate(train, steps=None, verbose=1)

### CategoricalCrossentropy
x,y = data_gen_all(num=20, in_shape=INPUT_SHAPE, num_labels=3)
print(x.shape)
model.fit(x=x, y=y, epochs=EPOCHS, verbose=2)

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import time as tm

INPUT_SHAPE=[3, 5]
NUM_POINTS=20
BATCH_SIZE=7
EPOCHS=4

def data_gen(num, in_shape):
    for i in range(num):
        x = np.random.rand(in_shape[0], in_shape[1])
        y = random.randint(0,2)
        yield x, y

train = tf.data.Dataset.from_generator(
    generator=data_gen,
    output_types=(tf.float32, tf.int32),
    output_shapes=([None, INPUT_SHAPE[1]],()),
    args=([NUM_POINTS, INPUT_SHAPE])
)

def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation="tanh", input_shape=input_shape),   
        tf.keras.layers.LSTM(1, activation="tanh"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    return model

model = create_model(input_shape=INPUT_SHAPE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0),
    loss= tf.keras.losses.SparseCategoricalCrossentropy()
    )
print(model.summary())
model.fit(train.batch(BATCH_SIZE), epochs=EPOCHS, verbose=2)
model.evaluate(train, steps=None, verbose=1)