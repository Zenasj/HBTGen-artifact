import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# on cpu nan at 1098 iter
# Iter    0: loss 9.99180138e-01,  runtime:     0.11
# Iter    1: loss 9.98347044e-01,  runtime:     0.11
# Iter    2: loss 9.97434497e-01,  runtime:     0.12
# on gpu nan at 2212 iter
# Iter    0: loss 9.99180079e-01,  runtime:     2.46
# Iter    1: loss 9.98346925e-01,  runtime:     2.47
# Iter    2: loss 9.97434497e-01,  runtime:     2.48

np.random.seed(1)
tf.keras.backend.set_floatx('float32')

func = lambda x: x
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

data = np.array(
    [[0., 0.07179281], [0., 0.44064897], [0., 0.7666122], [0., -0.655319],
     [0., -0.28546047], [0., 0.8460491], [0., 0.14823522], [0., -0.14381762],
     [0., 0.7200559], [0., -0.92189044], [0., 0.37300184], [0., -0.525946],
     [0., 0.07766213], [0., 0.370439], [0., 0.17311008], [0., 0.88918954],
     [0., -0.5910955], [0., -0.947578], [0., -0.7192261], [0., 0.5109261],
     [0., 0.85887444], [0., -0.75145805], [0., 0.89897853], [0., 0.23428982],
     [0., 0.5785587], [0., 0.0541162], [0., 0.97772217], [0., 0.24339144],
     [0., -0.72505057], [0., -0.39533487], [0., 0.6692513], [0., -0.7257285],
     [0., 0.93652314], [0., 0.17861107], [0., 0.38464522], [0., 0.38880032],
     [0., -0.73994285], [0., -0.9602397], [0., 0.07763347], [0., 0.6147826],
     [0., 0.68406177], [0., 0.39951673], [0., -0.17188802], [0., -0.10017573],
     [0., 0.7917724], [0., 0.35767105], [0., 0.7892133], [0., -0.62747955],
     [0., 0.7562349], [0., -0.16161098], [0., -0.77050805], [0., 0.8068038],
     [0., -0.37315163], [0., -0.3467102], [0., -0.70654285], [0., -0.8679997],
     [0., 0.5002886], [0., -0.7214473], [0., 0.7718842], [0., -0.5767438],
     [0., 0.8550172], [0., 0.4230495], [0., -0.7064882], [0., 0.11737966],
     [0., 0.326883], [0., -0.439112], [0., -0.99425936], [0., -0.94338703],
     [0., -0.8153228], [0., 0.8651909], [0., -0.96342343], [0., 0.9296801],
     [0., -0.50757784], [0., 0.24734442], [0., 0.80675906], [0., 0.38375422],
     [0., -0.7953311], [0., -0.4127717], [0., 0.39363632], [0., -0.30887854],
     [0., -0.8299116], [0., -0.603797], [0., -0.9452248], [0., -0.80330634],
     [0., 0.34093502], [0., -0.793548], [0., 0.6014891], [0., 0.7527783],
     [0., 0.38179383], [0., -0.9000931], [0., 0.4963313], [0., 0.45199597],
     [0., -0.9612661], [0., -0.30446827], [0., 0.9946457], [0., 0.14735897],
     [0., 0.24672022], [0., -0.20646505], [0., -0.20464632], [0., -0.1837264],
     [0., 0.8170703], [0., -0.15778475], [0., 0.5018849], [0., -0.8932749],
     [0., 0.10564396], [0., 0.91577905], [0., -0.01685368], [0., -0.42444932],
     [0., -0.30220333], [0., -0.46014422], [0., -0.99977124], [0., 0.06633057],
     [0., 0.15677923], [0., -0.46890667], [0., -0.36896873], [0., -0.6692916],
     [0., -0.17164145], [0., 0.756285], [0., -0.16595599], [0., 0.817191],
     [0., 0.5016242], [0., 0.3275893], [0., 0.50775236], [0., 0.02977822],
     [0., -0.10421295], [0., -0.9683575], [0., -0.6603392], [0., -0.1653904]],
    dtype=np.float32)
data_x = data[:, 0:1]
data_y = data[:, 1:2]


def loss_func(x, y):
    return tf.reduce_mean(tf.norm(func(x) - y, axis=1) / tf.norm(y, axis=1))


class MyNN(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.input_dims = [1, 1]
        self.func = func
        self.optimizer = optimizer

        self.net1 = tf.keras.layers.Dense(
            **{"units": 4, "activation": 'relu',
               "kernel_initializer": {
                   'class_name': 'glorot_uniform',
                   'config': {'seed': 1}}}
        )
        self.net2 = tf.keras.layers.Dense(
            **{"units": 1, "activation": None,
               "kernel_initializer": {
                   'class_name': 'glorot_uniform',
                   'config': {'seed': 1}}}
        )

    def train_one_step(self, x, y):
        with tf.GradientTape() as tape:
            x_pred = self(x, y)
            loss = loss_func(x_pred, y)
        grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, start_time=time.time(), max_iter=3000):
        for it in range(max_iter):
            loss = self.train_one_step(data_x, data_y)

            print("Iter %4d: loss %14.8e,  runtime: %8.2f"
                  % (it, loss.numpy(), time.time() - start_time))

    def call(self, x, y):
        r = y - self.func(x)
        g = self.net2(self.net1(r)) * 2e-3
        return x + g


model = MyNN()
model.train()

import numpy as np
import tensorflow as tf
import time
np.random.seed(1)
tf.keras.backend.set_floatx('float32')

func = lambda x: x
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

data = np.array(
    [[0., 0.07179281], [0., 0.44064897], [0., 0.7666122], [0., -0.655319],
     [0., -0.28546047], [0., 0.8460491], [0., 0.14823522], [0., -0.14381762],
     [0., 0.7200559], [0., -0.92189044], [0., 0.37300184], [0., -0.525946],
     [0., 0.07766213], [0., 0.370439], [0., 0.17311008], [0., 0.88918954],
     [0., -0.5910955], [0., -0.947578], [0., -0.7192261], [0., 0.5109261],
     [0., 0.85887444], [0., -0.75145805], [0., 0.89897853], [0., 0.23428982],
     [0., 0.5785587], [0., 0.0541162], [0., 0.97772217], [0., 0.24339144],
     [0., -0.72505057], [0., -0.39533487], [0., 0.6692513], [0., -0.7257285],
     [0., 0.93652314], [0., 0.17861107], [0., 0.38464522], [0., 0.38880032],
     [0., -0.73994285], [0., -0.9602397], [0., 0.07763347], [0., 0.6147826],
     [0., 0.68406177], [0., 0.39951673], [0., -0.17188802], [0., -0.10017573],
     [0., 0.7917724], [0., 0.35767105], [0., 0.7892133], [0., -0.62747955],
     [0., 0.7562349], [0., -0.16161098], [0., -0.77050805], [0., 0.8068038],
     [0., -0.37315163], [0., -0.3467102], [0., -0.70654285], [0., -0.8679997],
     [0., 0.5002886], [0., -0.7214473], [0., 0.7718842], [0., -0.5767438],
     [0., 0.8550172], [0., 0.4230495], [0., -0.7064882], [0., 0.11737966],
     [0., 0.326883], [0., -0.439112], [0., -0.99425936], [0., -0.94338703],
     [0., -0.8153228], [0., 0.8651909], [0., -0.96342343], [0., 0.9296801],
     [0., -0.50757784], [0., 0.24734442], [0., 0.80675906], [0., 0.38375422],
     [0., -0.7953311], [0., -0.4127717], [0., 0.39363632], [0., -0.30887854],
     [0., -0.8299116], [0., -0.603797], [0., -0.9452248], [0., -0.80330634],
     [0., 0.34093502], [0., -0.793548], [0., 0.6014891], [0., 0.7527783],
     [0., 0.38179383], [0., -0.9000931], [0., 0.4963313], [0., 0.45199597],
     [0., -0.9612661], [0., -0.30446827], [0., 0.9946457], [0., 0.14735897],
     [0., 0.24672022], [0., -0.20646505], [0., -0.20464632], [0., -0.1837264],
     [0., 0.8170703], [0., -0.15778475], [0., 0.5018849], [0., -0.8932749],
     [0., 0.10564396], [0., 0.91577905], [0., -0.01685368], [0., -0.42444932],
     [0., -0.30220333], [0., -0.46014422], [0., -0.99977124], [0., 0.06633057],
     [0., 0.15677923], [0., -0.46890667], [0., -0.36896873], [0., -0.6692916],
     [0., -0.17164145], [0., 0.756285], [0., -0.16595599], [0., 0.817191],
     [0., 0.5016242], [0., 0.3275893], [0., 0.50775236], [0., 0.02977822],
     [0., -0.10421295], [0., -0.9683575], [0., -0.6603392], [0., -0.1653904]],
    dtype=np.float32)
data_x = data[:, 0:1]
data_y = data[:, 1:2]

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x,y, training=training)

  return loss_func(y, y_)


def loss_func(x, y):
    return tf.reduce_mean(tf.norm(func(x) - y, axis=1) / tf.norm(y, axis=1))


class MyNN(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.input_dims = [1, 1]
        self.func = func
        self.optimizer = optimizer

        self.net1 = tf.keras.layers.Dense(
            **{"units": 4, "activation": 'relu',
               "kernel_initializer": {
                   'class_name': 'glorot_uniform',
                   'config': {'seed': 1}}}
        )
        self.net2 = tf.keras.layers.Dense(
            **{"units": 1, "activation": None,
               "kernel_initializer": {
                   'class_name': 'glorot_uniform',
                   'config': {'seed': 1}}}
        )
    def grad(self,inputs, targets):
      with tf.GradientTape() as tape:
        loss_value = loss(self, inputs, targets, training=True)
      return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train_one_step(self, x, y):
      loss_value, grads = self.grad(x, y)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      return loss_value


    def train(self, start_time=time.time(), max_iter=3000):
        for it in range(max_iter):
            loss = self.train_one_step(data_x, data_y)
            print("Iter %4d: loss %14.8e,  runtime: %8.2f" % (it, loss.numpy(), time.time() - start_time))

    def call(self, x, y):
        r = y - self.func(x)
        g = self.net2(self.net1(r)) * 2e-3
        return x + g


model = MyNN()
model.train()