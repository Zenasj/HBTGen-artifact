from tensorflow import keras
from tensorflow.keras import optimizers

from keras_contrib.applications.densenet import DenseNetImageNet161
import keras.optimizers
import tensorflow as tf

target_img_size = (224, 224, 3)
model = DenseNetImageNet161(input_shape=target_img_size)
model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='/path/to/dir')