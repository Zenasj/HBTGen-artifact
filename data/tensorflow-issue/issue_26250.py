import tensorflow as tf
from tensorflow import keras

model = Sequential()
model.add(LSTM (120,activation = "tanh", input_shape=(timesteps,dim), return_sequences=True))
model.add(LSTM(120, activation = "tanh", return_sequences=True))
model.add(LSTM(120, activation = "tanh", return_sequences=True))
model.add(LSTM(120, activation = "tanh", return_sequences=True))
model.add(LSTM(120, activation = "tanh", return_sequences=True))
model.add(LSTM(120, activation = "tanh", return_sequences=True))
model.add(Dense(dim))
model.compile(optimizer="adam", loss="mse",  metrics=[tf.keras.metrics.Precision()])

history = model.fit(data,data, 
                    epochs=100,
                    batch_size=10,
                    validation_split=0.2,
                    shuffle=True,
                    callbacks=[ch]).history

model = Sequential()
model.add(LSTM (120,activation = "tanh", input_shape=(timesteps,dim), return_sequences=True))
model.add(LSTM(120, return_sequences=True))
model.add(LSTM(120, return_sequences=True))
model.add(LSTM(120, return_sequences=True))
model.add(LSTM(120, return_sequences=True))
model.add(LSTM(120, return_sequences=True))
model.add(Dense(dim, activation="softmax"))