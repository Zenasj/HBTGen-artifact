import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class Model:
    def __init__(self):
        tf.keras.backend.clear_session()
        
        inp = tf.keras.layers.Input((32,32,3), name='input_image')
        net = tf.keras.layers.Conv2D(3, 3, padding="same")(inp)

        self.model  = tf.keras.Model(inputs=inp, outputs=net)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_obj = tf.keras.losses.MeanSquaredError()
        
    @tf.function
    def train_step(self):        
        angles_rad = tf.random.uniform((), 0, 3.14)
        images = tf.random.uniform((2,32,32,3))

        with tf.GradientTape() as tape:
            features = self.model(images, training=True)

            #angles_rad = tf.constant([0.5,0.4])
            rot_features = tfa.image.rotate(features, angles_rad, interpolation='NEAREST', name="rotate_features")
            
            loss = self.loss_obj(images, rot_features)

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss
    
    def train(self, nr_epochs):
        for epoch in range(nr_epochs):
            loss = self.train_step()
            print("Train Epoch {}: {}".format(epoch, loss))

trainer = Model()
trainer.train(5000)