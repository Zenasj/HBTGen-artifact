from tensorflow import keras

bottleneck.tflite
inference.tflite
initialize.tflite
optimizer.tflite
train_head.tflite

import tensorflow as tf
from google.colab import drive
drive.mount('/content/drive')
nb_path = '/content/drive/MyDrive/ML/converter/'
sys.path.insert(0,nb_path)
from tfltransfer import bases
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter
base = bases.MobileNetV2Base(image_size=224)

head = tf.keras.Sequential([
layers.Flatten(input_shape=(7, 7, 1280)),
layers.Dense(
units=32,
activation='relu',
kernel_regularizer=l2(0.01),
bias_regularizer=l2(0.01)),
layers.Dense(
units=4,
activation='softmax',
kernel_regularizer=l2(0.01),
bias_regularizer=l2(0.01)),
])
head.compile(loss='categorical_crossentropy', optimizer='sgd')
converter = TFLiteTransferConverter(4,
base,
heads.KerasModelHead(head),
optimizers.SGD(3e-2),
train_batch_size=20)
converter.convert_and_save('custom_keras_model')