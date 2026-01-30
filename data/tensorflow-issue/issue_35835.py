import tensorflow as tf
from tensorflow import keras

model = tf.keras.Model(sent_input, preds)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.load_weights(...)
while True:
    inputs = prepare_inputs(x)
    model.predict(inputs)