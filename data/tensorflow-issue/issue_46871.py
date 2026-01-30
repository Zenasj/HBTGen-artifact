import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

for i in ls:
        pretrained_model = efficient_net[i]
        x = pretrained_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output[i] = tf.keras.layers.Dense(CLASSES,activation="sigmoid", dtype='float32')(x)


#You can use the name parameter in layer function. It is best if you do them seperately and not in a for loop.