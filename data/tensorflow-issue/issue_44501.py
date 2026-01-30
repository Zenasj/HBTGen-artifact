import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(90),
])

datagen = ImageDataGenerator(
    rotation_range=model_params['rotation_range'],  # randomly rotate images in the range (degrees, 0 to 180)  
    horizontal_flip=model_params['horizontal_flip'],  # randomly flip images
    vertical_flip=model_params['vertical_flip'],

)