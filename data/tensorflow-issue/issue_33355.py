from tensorflow.keras import optimizers

keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix)

def create_simple_model(input_dim, rnn_units=None, learning_rate=DEFAULT_LR):
    RMSprop = keras.optimizers.RMSprop

    rnn_units = rnn_units or RNN_UNITS
    model = keras.Sequential()
    model.add(layers.LSTM(rnn_units, input_shape=input_dim))
    model.add(layers.Dense(input_dim[-1], activation='softmax'))

    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model