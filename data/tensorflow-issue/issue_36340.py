import math
import random
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from matplotlib import pyplot

class GanPointGraph_Keras(object):
    
    def __init__(self):
        self.latent_dim = 5
        self.discriminator = self.define_discriminator()
        self.generator = self.define_generator(self.latent_dim)
        self.gan_model = self.define_gan(self.generator, self.discriminator)

    def define_discriminator(self, n_inputs=2):
        model = Sequential()
        model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(K.eval(model.optimizer.lr))
        return model
    
    def define_generator(self, latent_dim, n_outputs=2):
        model = Sequential()
        model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
        model.add(Dense(n_outputs, activation='linear'))
        return model
    
    def define_gan(self, generator, discriminator):
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model
    
    def generate_latent_points(self, n):
        x_input = randn(self.latent_dim * n)
        x_input = x_input.reshape(n, self.latent_dim)
        return x_input
    
    def generate_fake_samples(self, n):
        x_input = self.generate_latent_points(n)
        X = self.generator.predict(x_input)
        return X

    def generate_real_samples(self, n):
        X1 = rand(n) - 0.5
        X2 = X1 * X1
        X1 = X1.reshape(n, 1)
        X2 = X2.reshape(n, 1)
        X = hstack((X1, X2))    
        return X
    
    def train(self):
        n_batch = 128
        half_batch = int(n_batch / 2)
        x_real = self.generate_real_samples(half_batch)
        y_real = ones((half_batch, 1))
        x_fake = self.generate_fake_samples(half_batch)
        y_fake = zeros((half_batch, 1))
        self.discriminator.train_on_batch(x_real, y_real)
        self.discriminator.train_on_batch(x_fake, y_fake)
        x_gan = self.generate_latent_points(n_batch)
        y_gan = ones((n_batch, 1))
        self.gan_model.train_on_batch(x_gan, y_gan)

if __name__ == "__main__":
    g = GanPointGraph_Keras();

    for epoch in range(10000):
        print('Epoch', epoch)
        g.train()
        if epoch % 1000 == 0:
            g_objects = g.generate_fake_samples(100)
            r_objects = g.generate_real_samples(100)
 
            pyplot.clf()
            pyplot.title('Keras iteration ' + str(epoch))
            pyplot.scatter([i[0] for i in r_objects], [i[1] for i in r_objects], c='black')
            pyplot.scatter([i[0] for i in g_objects], [i[1] for i in g_objects], c='red')
            pyplot.show()

import tensorflow as tf
tf.enable_eager_execution() # if using TF 1.15.x
from tensorflow.keras import layers

import numpy as np
from numpy.random import rand
from numpy import hstack

from matplotlib import pyplot

class GanPointGraph(object):

    def __init__(self):
        self.latent_dim = 5
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()
        
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)        
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    def make_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(15, activation='relu', input_dim=self.latent_dim))
        model.add(layers.Dense(2))
        return model
      
    def make_discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(25, activation='relu', input_dim=2))
        model.add(layers.Dense(1, activation='sigmoid')) # (-infinity, infinity) -> (0, 1)
        return model
    
    def generator_loss(self, fake_output):
        #return self.cross_entropy(tf.ones_like(fake_output), fake_output)
        return tf.reduce_mean(tf.math.log(1-fake_output))

    def discriminator_loss(self, real_output, fake_output):
        #real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        #fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        #total_loss = real_loss + fake_loss
        #return total_loss
        loss_real = tf.reduce_mean(-tf.math.log(real_output))
        loss_fake = tf.reduce_mean(-tf.math.log(1-fake_output))
        D_loss = loss_real + loss_fake
        return D_loss

    def generate_real_samples(self, n):
        X1 = rand(n) - 0.5
        X2 = X1 * X1
        X1 = X1.reshape(n, 1)
        X2 = X2.reshape(n, 1)
        x_train = hstack((X1, X2))
        return x_train
    
    def generate_fake_samples(self, n):
        z_sample = np.random.normal(0, 1.0, size=[n, self.latent_dim]).astype(np.float32)
        return self.generator(z_sample, training=False).numpy()
    
    def train(self):
        images = self.generate_real_samples(128);
        noise = tf.random.normal([images.shape[0], self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    
if __name__ == "__main__":
    g = GanPointGraph();
    
    for epoch in range(10000):
        print('Epoch', epoch)
        g.train()
        if epoch % 1000 == 0:
            g_objects = g.generate_fake_samples(100)
            r_objects = g.generate_real_samples(100)
 
            pyplot.clf()
            pyplot.title('Tensorflow iteration ' + str(epoch))
            pyplot.scatter([i[0] for i in r_objects], [i[1] for i in r_objects], c='black')
            pyplot.scatter([i[0] for i in g_objects], [i[1] for i in g_objects], c='red')
            pyplot.show()