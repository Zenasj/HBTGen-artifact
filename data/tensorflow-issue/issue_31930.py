from tensorflow.keras import layers
from tensorflow.keras import models

last_layer = Dense(1, activation='sigmoid')(previous_layer)

last_layer = Dense(1)(previous_layer)
last_layer = sigmoid(last_layer)

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import BinaryCrossentropy
import numpy as np


x_train = np.array([[1000]])
y_train = [0]

inp = Input(x_train.shape[1])
out = Dense(1, trainable=False, use_bias=False)(inp)
model = Model(inp, out)
model.get_layer('dense').set_weights([np.array([[1]])])
model.compile(optimizer='adam', loss=BinaryCrossentropy(from_logits=True))

print('\nFrom logits=True, no activation')
model.fit(x_train, y_train, epochs=1)
print('Prediction:')
print(model.predict(x_train))


inp = Input(x_train.shape[1])
out = Dense(1, trainable=False, use_bias=False)(inp)
out = sigmoid(out)
model = Model(inp, out)
model.get_layer('dense_1').set_weights([np.array([[1]])])
model.compile(optimizer='adam', loss=BinaryCrossentropy(from_logits=False))

print('\nFrom logits=False, separate sigmoid')
model.fit(x_train, y_train, epochs=1)
print('Prediction:')
print(model.predict(x_train))


inp = Input(x_train.shape[1])
out = Dense(1, trainable=False, use_bias=False, activation='sigmoid')(inp)
model = Model(inp, out)
model.get_layer('dense_2').set_weights([np.array([[1]])])
model.compile(optimizer='adam', loss=BinaryCrossentropy(from_logits=False))

print('\nFrom logits=False, sigmoid in dense')
model.fit(x_train, y_train, epochs=1)
print('Prediction:')
print(model.predict(x_train))