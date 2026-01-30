model = Sequential()
model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',
                 input_shape=(time_window_size, 1)))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(64))
model.add(Dense(units=time_window_size, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metric])
model.compile(optimizer="sgd", loss="mse", metrics=[metric])