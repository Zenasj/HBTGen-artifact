import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

"""
Learn the very basics by running MNIST in TF 2.0
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers
import numpy as np
import datetime
L2_REG = 0.0000001
LEARN_RATE = 1e-5
NUM_LABELS = 10


class MyFirstConvnet(tf.keras.Model):
    def __init__(self):
        super(MyFirstConvnet, self).__init__()
        self.layer1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.layer2 = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.pool = layers.MaxPooling2D((2, 2))
        self.layer3 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.flatten = layers.Flatten()
        self.classifier = layers.Dense(NUM_LABELS, activation='softmax')
        self.batchnorm1 = layers.BatchNormalization(scale=False)
        self.batchnorm2 = layers.BatchNormalization(scale=False)

    def call(self, inputs):
        x = self.batchnorm1(self.layer1(inputs))
        x = self.layer2(x)
        tf.summary.scalar('layer_2_activation_sum', data=np.sum(x, axis=None))  # ERROR THROWN HERE
        x = self.batchnorm2(self.pool(x))
        x = self.layer3(x)
        x = self.flatten(x)
        return self.classifier(x)


if __name__ == '__main__':
    # load up MNIST
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype(np.float32)/255.0
    test_images = test_images.reshape((10000, 28, 28, 1)).astype(np.float32)/255.0

    # model must be 'compiled' which integrates information about training and stores it in the model structure
    model = MyFirstConvnet()
    optimizer = tf.keras.optimizers.Adam(lr=LEARN_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # set up tensorboard
    logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # train
    model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.05, shuffle=True,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True),
                         tensorboard_callback])

    # test
    model.evaluate(test_images, test_labels, verbose=1)

def call(self, inputs):
        x = self.batchnorm1(self.layer1(inputs))
        x = self.layer2(x)
        print(x)
        print(np.sum(x))
        tf.summary.scalar('layer_2_activation_sum', data=np.sum(x, axis=None))  # ERROR THROWN HERE
        x = self.batchnorm2(self.pool(x))
        x = self.layer3(x)
        x = self.flatten(x)
        return self.classifier(x)

const = tf.constant([3,3], dtype=tf.float32)
np.sum(const)

tf.summary.scalar('layer_2_activation_sum', data=np.sum(x, axis=None))  # ERROR THROWN HERE

import numpy as np
import tensorflow as tf

tf.enable_v2_behavior()
print("We're in eager mode:   ", tf.random.uniform(shape=(4,)))

class LoggingIdentityLayer(tf.keras.layers.Layer):
  """A keras layer that prints some stuff."""
  
  def call(self, inputs):
    print("Inputs to {}: {}".format(self.name, tf.squeeze(inputs)))

    return tf.identity(inputs)
    # The base Layer is an identity layer
    return super(LoggingIdentityLayer, self).call(inputs=inputs)

  def compute_output_shape(self, input_shape):
    return input_shape  # This is an identity layer

x=np.random.random(size=(16, 1)),
y=np.random.random(size=(16, 1))

inp = tf.keras.layers.Input(shape=(1,))
dense_results = tf.keras.layers.Dense(1)(inp)
log_layer = LoggingIdentityLayer()(dense_results)
model = tf.keras.models.Model(inp, log_layer)

model.compile(loss="mse", optimizer="sgd")
model.fit(x=x, y=x, batch_size=4)
print()

# Run the entire model in eager mode:
model.compile(loss="mse", optimizer="sgd",
              run_eagerly=True)
model.fit(x=x, y=x, batch_size=4)
print()

# Only make a certain layer dynamic
dynamic_log_layer = LoggingIdentityLayer(dynamic=True)(dense_results)
model_prime = tf.keras.models.Model(inp, dynamic_log_layer)
model_prime.compile(loss="mse", optimizer="sgd")
model_prime.fit(x=x, y=x, batch_size=4)