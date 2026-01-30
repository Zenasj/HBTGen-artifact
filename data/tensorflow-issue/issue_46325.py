import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class batchNormalization(tf.keras.layers.Layer):
    def __init__(self, shape, Trainable, **kwargs):
        super(batchNormalization, self).__init__(**kwargs)
        self.shape = shape
        self.Trainable = Trainable
        self.beta = tf.Variable(initial_value=tf.zeros(shape), trainable=Trainable)
        self.gamma = tf.Variable(initial_value=tf.ones(shape), trainable=Trainable)
        self.moving_mean = tf.Variable(initial_value=tf.zeros(self.shape), trainable=False)
        self.moving_var = tf.Variable(initial_value=tf.ones(self.shape), trainable=False)

    def update_var(self,inputs):
        wu, sigma = tf.nn.moments(inputs, axes=[0, 1, 2], shift=None, keepdims=False, name=None)
        var = tf.math.sqrt(sigma)
        self.moving_mean = self.moving_mean * 0.09 + wu * 0.01
        self.moving_var = self.moving_var * 0.09 + var * 0.01
        return wu,var

    def call(self, inputs):
        wu, var = self.update_var(inputs)
        return tf.nn.batch_normalization(inputs, wu, var, self.beta,
                                         self.gamma, variance_epsilon=0.001)


@tf.function
def train_step(model, inputs, label,optimizer):
    with tf.GradientTape(persistent=False) as tape:
        predictions = model(inputs, training=1)
        loss = tf.keras.losses.mean_squared_error(predictions,label)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


if __name__=='__main__':
    f=tf.ones([2,256,256,8])
    label=tf.ones([2,256,256,8])
    inputs = tf.keras.Input(shape=(256,256,8))
    outputs=batchNormalization([8],True)(inputs)
    Model = tf.keras.Model(inputs=inputs, outputs=outputs)
    Layer = batchNormalization([8],True)
    print(len(Model.variables))
    print(len(Model.trainable_variables))
    print(len(Layer.variables))
    print(len(Layer.trainable_variables))
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    for i in range(0,100):
        train_step(Layer, f, label,optimizer)
        # train_step(Model,f,label,optimizer)

#self.moving_mean = self.moving_mean * 0.09 + wu * 0.01
#self.moving_var = self.moving_var * 0.09 + var * 0.01

self.moving_mean.assign(self.moving_mean * 0.09 + wu * 0.01)
self.moving_var.assign(self.moving_var * 0.09 + var * 0.01)