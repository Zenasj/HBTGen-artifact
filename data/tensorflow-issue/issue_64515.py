from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

def lenet5():
    model = tf.keras.Sequential()
    couche0 = tf.keras.layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28,1))
    couche1 = tf.keras.layers.MaxPooling2D((2, 2))
    couche2 = tf.keras.layers.Conv2D(16, activation='relu',kernel_size=(3, 3))
    couche3 = tf.keras.layers.MaxPooling2D((2, 2))
    couche4 = tf.keras.layers.Flatten()
    couche5 = tf.keras.layers.Dense(120, activation='relu')
    couche6 = tf.keras.layers.Dense(84, activation='relu')
    couche7 = tf.keras.layers.Dense(10, activation='softmax', name="output")
    model.add(tf.keras.Input(shape=( 28,28, 1), name="digit", dtype=tf.float32))
    model.add(couche0)
    model.add(couche1)
    model.add(couche2)
    model.add(couche3)
    model.add(couche4)
    model.add(couche5)
    model.add(couche6)
    model.add(couche7)
    return model

model = lenet5()
model.compile(optimizer='adam',loss=tf.keras.losses.categorical_crossentropy,metrics=['CategoricalAccuracy'])

model.load_weights("mnist_0000062_.weights.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
# Save the converted model to a file
temp_model_file = "lenet5.tflite"
with open(temp_model_file,'wb') as f:
   f.write(tflite_model)