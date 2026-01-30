import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def hsv_conversion(x):
    return tf.image.rgb_to_hsv(x)

def create_model():
    layer1 = keras.Input((128,64,3))
    x = layers.Lambda(hsv_conversion)(layer1)
    x = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2),  padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(2, activation='softmax')(x)
    model = keras.Model(layer1, output)
    return model