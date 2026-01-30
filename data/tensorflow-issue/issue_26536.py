import numpy as np
import tensorflow as tf
from tensorflow import keras

# Model subclassing, error
class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__(name='Discriminator')
    self.conv1 = layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2),
                              padding='same')
    self.conv1_bn = layers.BatchNormalization()
    self.conv1_out = layers.LeakyReLU()
    
    self.conv2 = layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2),
                              padding='same')
    self.conv2_bn = layers.BatchNormalization()
    self.conv2_out = layers.LeakyReLU()
    
    self.conv3 = layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1),
                               padding='same')
    self.conv3_bn = layers.BatchNormalization()
    self.conv3_out = layers.LeakyReLU()

    self.flatten = layers.Flatten()
    # Error occurs here, code does run without dense layer
    self.output = layers.Dense(1)
    
  def call(self, input, training=True):
    conv1 = self.conv1(input)
    conv1_bn = self.conv1_bn(conv1)
    conv1 = self.conv1_out(conv1_bn)

    conv2 = self.conv2(conv1)
    conv2_bn = self.conv2_bn(conv2)
    conv2 = self.conv2_out(conv2_bn)
    
    conv3 = self.conv3(conv2)
    conv3_bn = self.conv3_bn(conv3)
    conv3 = self.conv3_out(conv3_bn)
    
    flatten = self.flatten(conv3)
    output = self.output(flatten)
    
    return output

# Sequential API, no error
def make_discriminator_sequentialmodel():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
     
    return model
  
discriminator_sequential = make_discriminator_sequentialmodel()
fake_input = np.ones((32, 28, 28, 1), dtype=np.float32)
print(discriminator_sequential(fake_input))

# Functional API, no error
def make_discriminator_functionalmodel():
  inputs = tf.keras.Input(shape=(28, 28, 1))
  
  x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
  x = layers.LeakyReLU()(x)
  x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(x)
  x = layers.LeakyReLU()(x)
  x = layers.Flatten()(x)
  x = layers.Dense(1)(x)
  
  model = tf.keras.Model(inputs=inputs, outputs=x)
  
  return model

discriminator_functional = make_discriminator_functionalmodel()
fake_input = np.ones((32, 28, 28, 1), dtype=np.float32)
print(discriminator_functional(fake_input))