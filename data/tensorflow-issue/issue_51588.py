import random
from tensorflow import keras
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

class GatherModel(tf.keras.Model):
    def __init__(self,ind1,w1):
        super(GatherModel, self).__init__()
        self.ind1=ind1
        self.w1=tf.cast(w1,tf.float32)
        self.lambda1 = tf.Variable(initial_value=tf.constant(0.1), trainable=True, name='lambda1')

    def __call__(self, inputs,training=0):
        y=inputs
        for i in range(5):
            y=tf.transpose(y, [1, 2, 3, 0])
            y=tf.gather_nd(y*1.0, self.ind1)
            y=y*self.w1
            y=tf.reduce_sum(y,0)
            y=tf.transpose(y,[3,0,1,2])
            y = self.lambda1*y
        return y

# @tf.function
def train_step(model, inputs, labels, Loss, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=1)
        loss = Loss(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

if __name__ == '__main__':
    ind1=np.random.randint(0,300,[367, 217, 721, 2])
    w1=np.random.normal(size=[367, 217, 721, 1, 1])
    Model=GatherModel(ind1,w1)
    inputs=tf.random.normal([2,256,256,1])
    labels= tf.random.normal([2,217,721,1])
    loss=tf.keras.losses.MeanSquaredError()
    optimizer=tf.keras.optimizers.Adam(0.001)
    for i in range(10):
        LL=train_step(Model,inputs,labels,loss,optimizer)
        print(LL)

if __name__ == '__main__':
    ind1=tf.random.uniform([367, 217, 721, 2], minval=0, maxval=300, dtype=tf.int32)
    w1=tf.random.normal([367, 217, 721, 1, 1])
    Model=GatherModel(ind1,w1)