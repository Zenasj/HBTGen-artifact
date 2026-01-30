import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False
maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(3, activation='sigmoid')
model = tf.keras.Sequential([
    base_model,
    maxpool_layer,
    prediction_layer
])