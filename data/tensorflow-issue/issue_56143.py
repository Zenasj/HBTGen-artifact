from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from tensorflow.python.keras import activations
from sklearn.model_selection import KFold
from tensorflow.python.keras.callbacks import EarlyStopping
def myconv2d(ix, w, padding):
   # filter shape: [filter_height, filter_width, in_channels, out_channels]
   # flatten filters
   filter_height = int(w.shape[0])
   filter_width = int(w.shape[1])
   in_channels = int(w.shape[2])
   out_channels = int(w.shape[3])
   ix_height = int(ix.shape[1])
   ix_width = int(ix.shape[2])
   ix_channels = int(ix.shape[3])
   filter_shape = [filter_height, filter_width, in_channels, out_channels]
   flat_w = tf.reshape(w, [filter_height * filter_width * in_channels, out_channels])
   patches = tf.image.extract_patches(
       ix,
       sizes=[1, filter_height, filter_width, 1],
       strides=[1, 1, 1, 1],
       rates=[1, 1, 1, 1],
       padding= padding
   )
   patches_reshaped = tf.reshape(patches, [-1, ix_height, ix_width, filter_height * filter_width * ix_channels])
   feature_maps = []
   for i in range(out_channels):
       feature_map = tf.reduce_sum(tf.multiply(flat_w[:, i], patches_reshaped), axis=3, keepdims=True)
       feature_maps.append(feature_map)
   features = tf.concat(feature_maps, axis=3)
   return features

class MyConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size,padding='SAME', **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        #self.units= units
        super(MyConv2D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "padding" : self.padding,
        })
        return config

    def build(self, input_shape):
        # only have a 3x3 kernel
        shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(
            name="bias", shape=(self.filters,), initializer="random_normal", trainable=True
        )
        super((MyConv2D, self).build(input_shape))

    def call(self, inputs):
        result = myconv2d(inputs, self.kernel, self.padding) + self.b
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)


def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = np_utils.to_categorical(trainY)
	testY = np_utils.to_categorical(testY)
	# convert from integers to floats
	train_norm = trainX.astype('float32')
	test_norm = testX.astype('float32')
	# normalize to range 0-1
	trainX = train_norm / 255.0
	testX = test_norm / 255.0
	# return normalized images
	return trainX, trainY, testX, testY


def create_model():
	# creating a sequantial model
	model = tf.keras.Sequential()
	# adding convolution2D layer to the model of 32 filters of size 3x3
	model.add(MyConv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)))
	model.add(tf.keras.layers.Activation(activations.relu))
	# adding a maxpooling 2D layer of size 2x2
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
	# adding a Flatten layer
	model.add(tf.keras.layers.Flatten())
	# adding Dense layer with 'relu' activation
	model.add(tf.keras.layers.Dense(100, activation='relu'))
	# adding Dense layer with 'softmax' activation for output
	model.add(tf.keras.layers.Dense(10, activation='softmax'))
	return model


def define_model(model):
	# compile model
	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


if __name__ == "__main__":
	train_images, train_labels, test_images, test_labels = load_dataset()

	model = create_model()

	# compile model
	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
	# fit model
	history = model.fit(train_images, train_labels, epochs=50, batch_size=32,
						validation_data=(test_images, test_labels), callbacks=[es])
	# evaluate model
	scores = model.evaluate(test_images, test_labels, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1] * 100))
	# stores scores

# These imports are required to load operators' definition.
import tensorflow_text as tf_text
import sentencepiece as spm

converter = tf.lite.TFLiteConverter.from_keras_model(your_model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
model_data = converter.convert()