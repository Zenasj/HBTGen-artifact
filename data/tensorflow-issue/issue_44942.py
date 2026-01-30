# Build the model
tf_model = Sequential()
tf_model.add(LSTM(
    units=32,
    input_shape=[window, 1],    
))
tf_model.add(Dense(units=1))
tf_model.compile()
tf_model.fit(rolling_x, rolling_y, epochs=100)