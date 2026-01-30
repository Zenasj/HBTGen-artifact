import tensorflow as tf

model = Sequential()
model.add(InputLayer(input_shape=(32,20,3)))
model.add(Conv2D(8, (3, 3)))
# model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(11, activation = "softmax"))

name = "conv2d"
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(name + ".tfl", "wb").write(tflite_model)

model.add(InputLayer(input_shape=(32,20,3)))
# model.add(Conv2D(8, (3, 3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(11, activation = "softmax"))