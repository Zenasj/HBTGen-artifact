from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tempfile
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, ZeroPadding2D
from tensorflow.keras.datasets import mnist
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import numpy as np
(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
X_train = (np.expand_dims(X_train,3))
X_test = (np.expand_dims(X_test,3))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
train_datagen =ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,#mean=0
    featurewise_std_normalization=True,#/std
    rotation_range=20,
    # preprocessing_function=my_random_crop
)

test_datagen=ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
)
train_datagen.fit(X_train)
test_datagen.fit(X_test)
# parser = argparse.ArgumentParser()
# parser.add_argument(
#      '-trtm', '--trtmodel', type=str, default='../F_TorrentNet.trt')
# parser.add_argument(
#         '-m', '--model', type=str, default='F_TorrentNet.pth')
# args = parser.parse_args()
class NoOpQuantizeConfig(quantize_config.QuantizeConfig):
    """QuantizeConfig which does not quantize any part of the layer."""

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}
quantize_config = NoOpQuantizeConfig()
def LeNet():
    inputs = Input(shape=(28, 28,1))
    # x = Reshape((28, 28, 1), input_shape=(28, 28))(inputs)
    x = ZeroPadding2D(padding=(2, 2))(inputs)

    h = Conv2D(6, 5, strides=(1, 1), activation='relu')(x)
    # h_nor = BatchNormalization()(h)
    h_nor = tfmot.quantization.keras.quantize_annotate_layer(
      BatchNormalization(), quantize_config=quantize_config)(h)
    h_max = MaxPooling2D(pool_size=(2, 2))(h_nor)

    h = Conv2D(16, 5, strides=(1, 1), activation='relu')(h_max)
    # h_nor = BatchNormalization()(h)
    h_nor = tfmot.quantization.keras.quantize_annotate_layer(
        BatchNormalization(), quantize_config=quantize_config)(h)
    h_max = MaxPooling2D(pool_size=(2, 2))(h_nor)

    h = Flatten()(h_max)

    d = Dense(120, activation='relu')(h)
    # d_nor = BatchNormalization()(d)
    d_nor=tfmot.quantization.keras.quantize_annotate_layer(
     BatchNormalization(), quantize_config=quantize_config)(d)
    d = Dense(84, activation='relu')(d_nor)
    # d_nor = BatchNormalization()(d)
    d_nor = tfmot.quantization.keras.quantize_annotate_layer(
        BatchNormalization(), quantize_config=quantize_config)(d)


    y = Dense(10, activation='softmax')(d_nor)

    model = Model(inputs=inputs, outputs = y)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def XMLP():
    inputs = Input(shape=(28, 28,1))
    h = Flatten()(inputs)

    # h = Dense(512, activation='relu',
    #           kernel_initializer=tf.keras.initializers.random_normal(mean=0, stddev=0.01))(h)
    # h_nor = BatchNormalization()(h)
    h=Dense(512,activation='linear')(h)
    # h_nor=BatchNormalization()(h)
    h_nor=tfmot.quantization.keras.quantize_annotate_layer(
    BatchNormalization(), quantize_config=quantize_config)(h)
    h =Dense(512,activation='relu')(h_nor)

    # h = Dense(512, activation='relu',
    #           kernel_initializer=tf.keras.initializers.random_normal(mean=0, stddev=0.01))(h_nor)
    # h_nor = BatchNormalization()(h)
    h = Dense(512, activation='linear')(h)
    # h_nor = BatchNormalization()(h)
    h_nor = tfmot.quantization.keras.quantize_annotate_layer(
        BatchNormalization(), quantize_config=quantize_config)(h)
    h = Dense(512, activation='relu')(h_nor)

    # h = Dense(512, activation='relu',
    #           kernel_initializer=tf.keras.initializers.random_normal(mean=0, stddev=0.01))(h_nor)
    # h_nor = BatchNormalization()(h)
    h = Dense(512, activation='linear')(h)
    # h_nor = BatchNormalization()(h)
    h_nor = tfmot.quantization.keras.quantize_annotate_layer(
        BatchNormalization(), quantize_config=quantize_config)(h)
    h = Dense(512, activation='relu')(h_nor)

    # h = Dense(256, activation='relu',
    #           kernel_initializer=tf.keras.initializers.random_normal(mean=0, stddev=0.01))(h_nor)
    # h_nor = BatchNormalization()(h)
    h = Dense(256, activation='linear')(h)
    # h_nor = BatchNormalization()(h)
    h_nor = tfmot.quantization.keras.quantize_annotate_layer(
        BatchNormalization(), quantize_config=quantize_config)(h)
    h = Dense(256, activation='relu')(h_nor)
    y = Dense(10, activation='softmax')(h)

    model = Model(inputs=inputs, outputs=y)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model=LeNet()
model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=50, shuffle=True),
                    steps_per_epoch=len(X_train) / 50, epochs=40)
tf.saved_model.save(model, 'mlp')
model.summary()
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope
with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig}):
    q_aware_model = tfmot.quantization.keras.quantize_model(model)
    q_aware_model.summary()
# `quantize_model` requires a recompile.
print("qat_____")
q_aware_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
# q_aware_model.fit(X_train, y_train, batch_size=50, epochs=40)
q_aware_model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=50, shuffle=True),
                    steps_per_epoch=len(X_train) / 50, epochs=40)
# q_aware_model.save('lenet-qat.h5')
tf.saved_model.save(q_aware_model, 'mlp-qat')
q_aware_model.summary()
_, baseline_model_accuracy = model.evaluate(test_datagen.flow(X_test,y_test,batch_size=50,shuffle=False),
 verbose=1)

_, q_aware_model_accuracy = q_aware_model.evaluate(test_datagen.flow(X_test,y_test,batch_size=50,shuffle=False),
   verbose=1)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
open("converted_Lenet-new.tflite", "wb").write(quantized_tflite_model )