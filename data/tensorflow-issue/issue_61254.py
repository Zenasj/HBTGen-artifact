from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow import keras


def gru(num_units=25, input_shape=10):
# gru input layer
    input_tensor = keras.Input(shape=input_shape)
# gru hidden layer
    x = keras.layers.Embedding(input_dim=100, output_dim=10, input_length=None)(input_tensor)
    x = keras.layers.GRU(units=32, dropout=0.7338014982069313, return_sequences=True)(x)
    x = keras.layers.ActivityRegularization(l1=-0.616784030867379, l2=-0.9646777799675004)(x)
# gru output layer
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=num_units, activation="relu")(x))
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == "__main__":
    gru().summary()