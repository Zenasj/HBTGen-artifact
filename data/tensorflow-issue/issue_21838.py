import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

class NALU(tf.keras.layers.Layer):
    def __init__(self, num_outputs, **kwargs):
        self.num_outputs = num_outputs
        super(NALU, self).__init__(**kwargs)
        
    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.num_outputs)).as_list()
        get = tf.keras.initializers.get
        self.W_ = self.add_variable("W_", shape=shape, initializer=get('glorot_uniform'))
        self.M_ = self.add_variable("M_", shape=shape, initializer=get('glorot_uniform'))
        self.GAM = self.add_variable("GAM", shape=shape, initializer=get('glorot_uniform')) # Gate add & multiply
        self.GM = self.add_variable("GM", shape=shape, initializer=get('glorot_uniform')) # Gate multiply
        
        super(NALU, self).build(input_shape)
    
    def call(self, x):
        gam = tf.sigmoid(tf.matmul(x, self.GAM)) # The gate
        gm = tf.sigmoid(tf.matmul(x, self.GM)) # The gate
        
        W = tf.tanh(self.W_) * tf.sigmoid(self.M_)
        add = tf.matmul(x, W)
        
        # represents positive multiplication
        m = tf.exp(
            tf.matmul(tf.log(tf.abs(x) + 1e-14), W)
        )
        
        mul = (-1 * m) * (1.0 - gm) + gm * m 
        
        y = gam * add + (1.0 - gam) * mul
        return y
    
    def compute_output_shape(self, input_shape):
        # --------------IMPORTANT LINE---------------------
        shape = tf.TensorShape(input_shape)
        #  shape = tf.TensorShape(input_shape).as_list() is required to make the function work
        # -------------------------------------------------
        shape[-1] = self.num_outputs
        return tf.TensorShape(shape)
    
    def get_config(self):
        base_config = super(NALU, self).get_config()
        base_config['num_outputs'] = self.num_outputs
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def nalu_model():
    inp = tf.keras.layers.Input(shape=(2,))
    out = NALU(1)(inp)
    
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model

import numpy as np

def create_train():
#     x_train = np.random.randint(-100, 100, size=(10000000, 2), dtype=np.int32)
    x_train = np.random.uniform(-100, 100, size=(10000000, 2))

    first_neg = np.copy(x_train)
    first_neg[:, 0] *= -1

    second_neg = np.copy(x_train)
    second_neg[:, 1] *= -1

    all_neg = x_train * -1

    x_train = np.append(x_train, first_neg)
    x_train = np.append(x_train, second_neg)
    x_train = np.append(x_train, all_neg)
    x_train = x_train.reshape((40000000, 2))

    com = x_train[:,::-1]
    x_train = np.append(x_train, com).reshape((80000000, 2))
    y_train = x_train[:, 0] * x_train[:, 1]
    
    return x_train, y_train

x_train, y_train = create_train()
x_test = np.random.randint(-1000, 1000, size=(10000, 2))
y_test = x_test[:, 0] * x_test[:, 1]

model = nalu_model()

model.compile(optimizer='RMSProp',
              loss='MSE',
              metrics=['accuracy', 'MAE'])

cb = [tf.keras.callbacks.TensorBoard(log_dir='./add_logs', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True),
      tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00000001, patience=5, verbose=2, mode='auto')]

model.fit(x_train, y_train, epochs=500, batch_size=256, shuffle=True, callbacks=cb, validation_split=0.4)
model.evaluate(x_test, y_test)