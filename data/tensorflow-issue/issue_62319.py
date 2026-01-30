from tensorflow.keras import layers

import pathlib
import keras

path = "\\\\localhost\\c$"
path = pathlib.Path(path)
path = path/'test'
path.mkdir(exist_ok=True, parents=True)

model = keras.Sequential()
model.add(keras.Input(shape=(16,)))
model.add(keras.layers.Dense(8))

model.save(path/'model', save_format='tf')