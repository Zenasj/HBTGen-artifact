from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,3), name='digits')
x = layers.GRU(64, activation='relu', name='GRU',dropout=0.1)(inputs)
x = layers.Dense(64, activation='relu', name='dense')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer')
model.summary()
model.save('model',save_format='tf')