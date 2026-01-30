import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def make_and_fit():
    inp = Input(shape=(1,))
    out = Dense(1, activation='sigmoid')(inp)
    model = Model(inputs=inp, outputs=out)
    auc = tf.keras.metrics.AUC()
    model.compile(loss='binary_crossentropy',
                  metrics=[auc])
    x = np.random.normal(size=10)
    y = np.random.normal(size=10) > 0
    xv = np.random.normal(size=10)
    yv = np.random.normal(size=10) > 0
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc',
                                                     patience=2)
    model.fit(x=x,
              y=y,
              validation_data=(xv, yv),
              epochs=2,
              verbose=1,
              callbacks=earlystopping)

for i in range(2):
    make_and_fit()

for i in range(2):
  tf.keras.backend.clear_session()
  make_and_fit()

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras

# Define the model
model = Sequential(
    [
        Dense(32, input_shape=(20,), activation="relu"),
        Dense(20, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[keras.metrics.AUC, keras.metrics.F1Score],
)

# Train the model
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    patience=2,
    mode="min",
)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    callbacks=early_stopping,
)

# Evaluate the model
loss, AUC, F1 = model.evaluate(X_test, y_test)
print("AUC:", AUC)
print("F1:", F1)