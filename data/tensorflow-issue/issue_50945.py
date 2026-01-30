model = Sequential()
model.add(CuDNNLSTM(256, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(256, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("tanh"))

# Compile the model.
model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
print("Compiled!")

# Train the model.
model.fit(train_x, train_y, epochs=epoch, batch_size=512, validation_split=0.05)