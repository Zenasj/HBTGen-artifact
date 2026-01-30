from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def loss_fun(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return loss

# Create a dataset
x = np.random.rand(10, 180, 320, 3).astype(np.float32)
y = np.random.rand(10, 1).astype(np.float32)
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)

# Create a model
base_model = tf.keras.applications.MobileNet(input_shape=(180, 320, 3), weights=None, include_top=False)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

for input, target in dataset:

    for iteration in range(400):
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            prediction = model(input, training=False)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fun(target, prediction)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.inputs)
        print(grads)  # output: [None]
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip(grads, model.inputs))

        print('Iteration {}'.format(iteration))

for input, target in dataset:
    image = tf.Variable(input[0])
    for iteration in range(400):
        with tf.GradientTape() as tape:
            tape.watch(image)
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            prediction = model(input, training=False)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fun(target, prediction)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, image)
        print(grads)  # output: [None]
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip([grads], [image]))

        print('Iteration {}'.format(iteration))

for input, target in dataset:
    image0 = tf.convert_to_tensor(input[0])
    image1 = tf.convert_to_tensor(input[1])

    image0_var = tf.Variable(image0)

    for iteration in range(400):
        with tf.GradientTape() as tape:
            tape.watch(image0)
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            # prediction = model(input, training=False)  # Logits for this minibatch
            prediction = model([input[0], input[1]])
            print('prediction: {}'.format(prediction))
            # Compute the loss value for this minibatch.
            # loss_value = loss_fun(target, prediction)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(prediction, [input[0]])
        print(grads)  # output: [None]
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer.apply_gradients(zip(grads, [image0_var]))
        #optimizer.apply_gradients(zip(grads, model.trainable_weights))

        print('Iteration {}'.format(iteration))

with tf.GradientTape() as tape:
    tape.watch(model.input)
    model_vals = model(v)

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

tf.random.set_seed(0)

model_in = keras.Input(shape=(2,))
model_d1 = layers.Dense(64, activation='relu')(model_in)
model_d2 = layers.Dense(64, activation='relu')(model_d1)
model_d3 = layers.Dense(1)(model_d2)
model = keras.Model(inputs=[model_in], outputs=[model_d3])
model.summary()

v = tf.Variable([[0.1, 0.2]], name='v_var')

####################################################
with tf.GradientTape() as tape:
    tape.watch(model.input)
    model_vals = model(v)
print(len(tape.watched_variables())) 
# 7      ( = (1 kernel + 1 bias) * (3 dense layers) + (1 v_var) )
print(model.input)
# Tensor("input_1:0", shape=(None, 2), dtype=float32)
model_grad = tape.gradient(model_vals, model.input)
print(model_grad)
# None
####################################################

####################################################
with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(model.input)
    model_vals = model(v)
print(len(tape.watched_variables())) 
# 0      (model.input isn't a variable so it is ignored)
####################################################

####################################################
with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(v)
    model_vals = model(v)
print(len(tape.watched_variables())) 
# 1
print(v)
# <tf.Variable 'v_var:0' shape=(1, 2) dtype=float32, numpy=array([[0.1, 0.2]], dtype=float32)>
model_grad = tape.gradient(model_vals, v)
print(model_grad)
# tf.Tensor([[-0.04392226 -0.06807809]], shape=(1, 2), dtype=float32)
####################################################

import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Embedding
from tensorflow.keras.models import Model

def get_model():
    input_layer = Input(shape=(24,), name="input_layer")
    ##i am initilizing randomly. But you can use predefined embeddings. 
    x_embedd = Embedding(input_dim=13732, output_dim=100, input_length=24, mask_zero=True, 
                        embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1, seed=23),
                         name="Embedding_layer")(input_layer)
    
    x_lstm = LSTM(units=20, activation='tanh', recurrent_activation='sigmoid', use_bias=True, 
                 kernel_initializer=tf.keras.initializers.glorot_uniform(seed=26),
                 recurrent_initializer=tf.keras.initializers.orthogonal(seed=54),
                 bias_initializer=tf.keras.initializers.zeros(), name="LSTM_layer")(x_embedd)
    
    x_out = Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=45),
                  name="output_layer")(x_lstm)
    
    basic_lstm_model = Model(inputs=input_layer, outputs=x_out, name="basic_lstm_model")
    
    return basic_lstm_model

basic_lstm_model = get_model()

temp_features = np.random.randint(low=1, high=13732, size=(10,24))

def get_gradient(model, x):
    x_tensor = tf.Variable(tf.convert_to_tensor(x, dtype=tf.float32))
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        loss = model(x_tensor)
    #print(tape.watched_variables())
    grads = tape.gradient(loss, x_tensor)
    return grads

print(get_gradient(basic_lstm_model, temp_features))

tape.gradient(loss, input)

tape.gradient(loss, model.input)

import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128),
  tf.keras.layers.ReLU(),
  tf.keras.layers.Dense(10)
])
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

images, labels = next(iter(train_ds))

with tf.GradientTape() as tape:
  tape.watch(model.layers[0].output)
  predictions = model(image)
  loss = loss_object(labels, predictions)

grads = tape.gradient(loss, model.layers[0].output)
print(grads)