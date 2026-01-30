from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Softmax

from tensorflow.keras.callbacks import EarlyStopping

x_train = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
y_train = np.array([0, 1, 1, 0])

x_val = np.array([[0, 0], [0, 1]])
y_val = np.array([1, 0])

model = keras.Sequential([
    Dense(20),
    Dense(1),
    Softmax()
])

model.compile(optimizer=SGD(), loss='mse')
model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=5,
          callbacks=[
              EarlyStopping(monitor='val_loss',
                            baseline=0.5,
                            patience=3,
                            restore_best_weights=True)
          ])