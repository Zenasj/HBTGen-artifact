from tensorflow.keras import layers

from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile(optimizer='adam', loss='mse')
loss = keras.losses.mse(model.input, model.output)
updates = model.optimizer.get_updates(params=model.trainable_weights, loss=loss)