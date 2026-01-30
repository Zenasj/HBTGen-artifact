from tensorflow.keras import layers
from tensorflow.keras import models

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf #tf.__version__ = '2.6.0'
# tf.compat.v1.disable_eager_execution()
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8] #X.shape = (768, 8)
y = dataset[:,8]

def customLoss(yTrue,yPred):
    x_tensor = tf.convert_to_tensor(model.input, dtype=tf.float32)
    x_tensor = tf.cast(x_tensor, tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        output = model(x_tensor)
    DyDX = t.gradient(output, x_tensor)    
    dy_t = DyDX[:, 5:6][0]
    R_pred=dy_t
    # loss_data = tf.reduce_mean(tf.square(yTrue - yPred), axis=-1)
    loss_PDE = tf.reduce_mean(tf.square(R_pred))
    return loss_PDE

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=customLoss, optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=15, batch_size=10)