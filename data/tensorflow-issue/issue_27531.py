from tensorflow.keras import layers

feature_layer = SequenceFeatures(columns, name=INPUT_NAME)

model = keras.Sequential([
    feature_layer,
    keras.layers.LSTM(1, return_sequence=True),
    keras.layers.LSTM(1),
    keras.layers.Dense(1, activation='relu', name=OUTPUT_NAME)
])

model.compile(optimizer='adam', loss='mse')
model.fit(DATASET, epochs=5)