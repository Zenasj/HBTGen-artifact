from tensorflow.keras import layers

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="sigmoid"),
    ]
)