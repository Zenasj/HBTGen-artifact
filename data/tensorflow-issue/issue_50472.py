import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import matplotlib.pyplot as plt
import os

def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10 * np.sin(2 * np.pi * (x)) + epsilon

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp

tfd = tfp.distributions

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

train_size = 32
noise = 1.0

X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
y = f(X, sigma=noise)
y_true = f(X, sigma=0.0)

plt.scatter(X, y, marker='+', label='Training data')
plt.plot(X, y_true, label='Truth')
plt.title('Noisy training data and ground truth')
plt.savefig('data.png')
plt.close()

batch_size = train_size
num_batches = train_size / batch_size
kl_weight = 1.0 / num_batches

x_in = Input(shape=(1,))

x = tfp.layers.DenseFlipout(20, activation='relu')(x_in)
x = tfp.layers.DenseFlipout(20, activation='relu')(x)
x = tfp.layers.DenseFlipout(1)(x)
model = Model(x_in, x)

from tensorflow.keras import callbacks, optimizers, utils

def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return K.sum(-dist.log_prob(y_obs)) 

kl = sum(model.losses)
loss = neg_log_likelihood + kl 

model.compile(loss=loss, optimizer=optimizers.Adam(lr=0.08), metrics=['mse'])
utils.plot_model(model, to_file = 'model_flipout.png',  show_shapes = True, show_layer_names = True, show_dtype = True, dpi = 600)
model.summary()
model.fit(X, y, batch_size=batch_size, epochs=1500, verbose=0);
model.save(f'flipout.h5')    

import tqdm

X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
y_pred_list = []

for i in tqdm.tqdm(range(500)):
    y_pred = model.predict(X_test)
    y_pred_list.append(y_pred)
#import ipdb; ipdb.set_trace()                
y_preds = np.concatenate(y_pred_list, axis = 1)
y_mean = np.mean(y_preds, axis = 1)
y_sigma = np.std(y_preds, axis = 1)

plt.plot(X_test, y_mean, 'r-', label = 'Predictive mean');
plt.plot(X, y_true, 'b-', label='Truth')

plt.scatter(X, y, marker = '+', label = 'Training data')
plt.fill_between(X_test.ravel(), 
                 y_mean + 2 * y_sigma, 
                 y_mean - 2 * y_sigma, 
                 alpha = 0.5, label='Epistemic uncertainty')
plt.title('Prediction')
plt.legend();
plt.savefig('result_flipout.png')
plt.close()