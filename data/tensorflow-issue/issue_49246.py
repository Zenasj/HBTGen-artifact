from tensorflow.keras import layers

import tensorflow.keras as keras

model = keras.Sequential()
model.add(keras.layers.Input(shape=(1,)))

model.compile()
model.summary()

model.fit([1, 2, 3, 4, 5], callbacks=[keras.callbacks.TensorBoard(update_freq='batch')])
model.compile()
model.fit([1, 2, 3, 4, 5])

model.fit([1, 2, 3, 4, 5], callbacks=[keras.callbacks.TensorBoard(update_freq=1)])
model.fit([1, 2, 3, 4, 5], callbacks=[keras.callbacks.TensorBoard(update_freq=2)])

model.fit([1, 2, 3, 4, 5], callbacks=[keras.callbacks.TensorBoard(update_freq=2)])
model.fit([1, 2, 3, 4, 5], callbacks=[keras.callbacks.TensorBoard(update_freq=1)])

model.fit([1, 2, 3, 4, 5], callbacks=[keras.callbacks.TensorBoard(update_freq=1)])
model.train_function=None
model.fit([1, 2, 3, 4, 5], callbacks=[keras.callbacks.TensorBoard(update_freq=1)])