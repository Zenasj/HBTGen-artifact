from tensorflow import keras
from tensorflow.keras import models

import sys
import keras
import random
import numpy as np
import tensorflow as tf
from keras import layers
from keras import optimizers

print("python version:", sys.version)
print("tensorflow version:", tf.__version__)
print("numpy version:", np.__version__)

def train_using_strategy(strategy, train_input, train_label, test_input):
    # Load model
    with strategy.scope():
        model = keras.models.load_model("./model.keras")
        optimizer = optimizers.SGD(learning_rate=10.0)
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

        model.compile(optimizer=optimizer, loss=loss)

    # Train for 1 step
    model.fit(
        train_input, train_label, verbose=0, shuffle=False, batch_size=2400)

    pred = model(test_input)
    pred = tf.nn.softmax(pred)
    return pred


# Set random seeds
seed = 54078027
random.seed(seed)
tf.random.set_seed(seed)

# Training data, batch_size=2400
train_input = tf.random.uniform(shape=(2400, 32, 32, 3))
train_label = tf.one_hot(tf.random.uniform(
    shape=(2400, ), minval=0, maxval=10, dtype=tf.int32), 10)
test_input = tf.random.uniform(shape=(1, 32, 32, 3))
test_label = tf.one_hot(tf.random.uniform(
    shape=(1, ), minval=0, maxval=10, dtype=tf.int32), 10)
        
# Original model
layer_0 = layers.Input(shape=(32, 32, 3,))
layer_1 = layers.Conv2D(
    filters=5,
    kernel_size=(13, 13),
    strides=(1, 1),
    padding="valid",
    data_format="channels_last",
    dilation_rate=(1, 1),
    activation="tanh",
    use_bias=True,
    kernel_initializer="random_uniform",
    bias_initializer="random_uniform")(layer_0)
layer_2 = layers.ReLU(max_value=0.08354582293069757)(layer_1)
layer_3 = layers.Flatten()(layer_2)
layer_4 = layers.Dense(
    units=10,
    activation="linear",
    use_bias=False,
    kernel_initializer="random_uniform",
    bias_initializer="random_uniform")(layer_3)
layer_5 = layers.Reshape((10,))(layer_4)

model = keras.Model(layer_0, layer_5)
model.summary()

model.load_weights("./tensorflow.h5")

# Alternatively, load the model directly
# model = keras.models.load_model("./tensorflow.h5")

keras.models.save_model(model, "./model.keras")

res_cpu = train_using_strategy(
    strategy=tf.distribute.MirroredStrategy(devices=["/CPU:0"]), 
    train_input=train_input, 
    train_label=train_label, 
    test_input=test_input)

res_gpu = train_using_strategy(
    strategy=tf.distribute.MirroredStrategy(devices=["/GPU:0"]), 
    train_input=train_input, 
    train_label=train_label, 
    test_input=test_input)

print("max diff:", np.max(np.abs(res_cpu - res_gpu)))
print("result on CPU device:", res_cpu)
print("result on GPU device:", res_gpu)