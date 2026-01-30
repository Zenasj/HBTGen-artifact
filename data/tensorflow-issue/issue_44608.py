import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_size = (299, 299)
IMG_SHAPE = (*img_size, 3)
base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')
base_model.trainable = False
preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input

data_augmentation = [
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, fill_mode="constant")  
]

data_augmentation = tf.keras.Sequential(data_augmentation)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

top_layers = []
top_layers.extend([
    tf.keras.layers.Dense(512, activation="relu", 
                          kernel_initializer="glorot_normal", 
                          bias_initializer="glorot_uniform"),
    tf.keras.layers.Dropout(0.2)
])

prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid", 
                                         kernel_initializer="glorot_normal", 
                                         bias_initializer="glorot_uniform")

inputs = tf.keras.Input(shape=(*img_size, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
for layer in top_layers:
    x = layer(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

(...)

model2 = tf.keras.Model(inputs=model.input, 
                        outputs=model.get_layer('inception_resnet_v2').output)

model2 = tf.keras.Model(inputs=model.input, 
                        outputs=model.get_layer('inception_resnet_v2').output)

model2 = tf.keras.Model(inputs=model.input, 
                        outputs=model.get_layer('inception_resnet_v2').get_output_at(0))