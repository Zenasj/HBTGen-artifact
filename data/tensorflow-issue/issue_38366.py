from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

N = 1000
x = tf.convert_to_tensor(np.linspace(0, 1, num=N), dtype=tf.float32)

with tf.GradientTape() as t:
    x_inp = keras.Input(shape=(1,), name='indvar')
    t.watch(x_inp)
    dense = layers.Dense(100, activation='tanh')(x_inp)
    dense = layers.Dense(100, activation='tanh')(dense)
    dense = layers.Dense(100, activation='tanh')(dense)
    dense = layers.Dense(100, activation='tanh')(dense)
    dense = layers.Dense(100, activation='tanh')(dense)
    dense = layers.Dense(100, activation='tanh')(dense)
    dense = layers.Dense(100, activation='tanh')(dense)
    yhat = layers.Dense(1, name='y')(dense)
    dyhat = t.gradient(yhat, x_inp)
    model = keras.Model(inputs=x_inp, outputs=[yhat, dyhat])
    opt=tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='MSE')
    model.summary()
    model.fit(x=x, y=[x, np.sin(10*x)], epochs=10000, batch_size=N)
    y, dy = model(x)
    plt.plot(x, np.array(y))
    plt.plot(x, np.array(dy))
    plt.show()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

N = 1000
x = np.linspace(0, 1, num=N)

with tf.GradientTape() as t:
    x_inp = keras.Input(shape=(1,), name='indvar')
    t.watch(x_inp)
    reg=tf.keras.regularizers.l2(l=0.01)
    dense = layers.Dense(100, activation='tanh', kernel_regularizer=reg)(x_inp)
    for i in range(2):
        dense = layers.Dense(100, activation='tanh', kernel_regularizer=reg)(dense)
    yhat = layers.Dense(1, name='y')(dense)
    dyhat = t.gradient(yhat, x_inp)
    model = keras.Model(inputs=x_inp, outputs=[yhat, dyhat])
    opt=tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=[lambda y, y_act: 0,'MSE'])
    model.summary()
    model.fit(x=x, y=[x, x*x], epochs=1, batch_size=N)  #with this line,  numerical derivative will differ from analytical
    y, dy = model(x)
    dy_num = np.gradient(np.array(y).squeeze(), x)
    plt.plot(x, y)
    plt.plot(x, dy)
    plt.plot(x, dy_num)
    plt.legend(['y', 'dy', 'dynum'])
    plt.show()