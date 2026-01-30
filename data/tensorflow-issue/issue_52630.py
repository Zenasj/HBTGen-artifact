import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

quantize_model = tfmot.quantization.keras.quantize_model

net_input = Input(shape=(256,256,3))

model_output = U2NET(net_input)

model = Model(inputs = net_input, outputs = model_output)

qa_model = quantize_model(model)
lr = 1e-3

opt = tf.keras.optimizers.Adam(learning_rate = lr)

bce = BinaryCrossentropy()

qa_model.compile(optimizer = opt, loss = loss, metrics = None)

#quantize_model = tfmot.quantization.keras.quantize_model

net_input = Input(shape=(256,256,3))

model_output = U2NET(net_input)

model = Model(inputs = net_input, outputs = model_output)

#qa_model = quantize_model(model)
lr = 1e-3

opt = tf.keras.optimizers.Adam(learning_rate = lr)

bce = BinaryCrossentropy()

#qa_model.compile(optimizer = opt, loss = loss, metrics = None)

model.compile(optimizer = opt, loss = loss, metrics = None)

number_of_classes = 10

inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)

layer = keras.layers.Dense(128, activation="relu")(embedding)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation="relu")(layer)
outputs = keras.layers.Dense(number_of_classes, activation="softmax")(layer)

model = keras.Model(inputs, outputs)
quant_aware_model = tfmot.quantization.keras.quantize_model(model)

model.summary()