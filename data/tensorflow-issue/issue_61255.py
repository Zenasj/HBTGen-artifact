from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow import keras


def squeezenet(label_num=1000, input_shape=(224, 224, 3)):
# squeezenet input layer
    input_tensor = keras.Input(shape=input_shape)
# squeezenet hidden layer
    x = keras.layers.Conv2D(filters=96, activation="relu", kernel_size=(7,7), strides=2, padding="valid")(input_tensor)
    x = keras.layers.Cropping2D(cropping=-100)(x)
# squeezenet output layer
    output_tensor = keras.layers.Flatten()(keras.layers.Dense(units=label_num, activation="softmax")(x))
    model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model


if __name__ == "__main__":
    squeezenet().summary()