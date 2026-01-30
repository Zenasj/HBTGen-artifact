import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

inputs = tf.keras.Input(shape=(784,), name='digits')
#x = tf.keras.layers.Activation('relu')(inputs)
x = tf.keras.activations.relu(inputs)
outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()

model.save('path_to_my_model.h5')