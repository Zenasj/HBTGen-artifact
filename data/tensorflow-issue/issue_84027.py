from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import keras
import numpy as np

class InterruptingCallback(keras.callbacks.Callback):
   def on_epoch_begin(self, epoch, logs=None):
     if epoch == 4:
       raise RuntimeError('Interrupting!')

callback = keras.callbacks.BackupAndRestore(backup_dir="/tmp/backup")

# Define the model with an explicit input shape
model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(20,)),  # Specify the input shape
    keras.layers.Dense(10)
])

model.compile(keras.optimizers.SGD(), loss='mse')

# Now the model is built before training
try:
    model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
              batch_size=1, callbacks=[callback, InterruptingCallback()],
              verbose=0)
except Exception as e:
    print(e)

history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=10, batch_size=1, callbacks=[callback],
                    verbose=0)

print(len(history.history['loss']))

import keras
import numpy as np

class InterruptingCallback(keras.callbacks.Callback):
   def on_epoch_begin(self, epoch, logs=None):
     if epoch == 4:
       raise RuntimeError('Interrupting!')

callback = keras.callbacks.BackupAndRestore(backup_dir="/tmp/backup")

# Define the model without specifying input shape
model = keras.models.Sequential([
    keras.layers.Dense(10)
])

model.compile(keras.optimizers.SGD(), loss='mse')

# Build the model by passing a batch of data
model.build(input_shape=(None, 20))  # Here, 20 is the number of features in your input data

# Now the model is built before calling fit
try:
    model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
              batch_size=1, callbacks=[callback, InterruptingCallback()],
              verbose=0)
except Exception as e:
    print(e)

history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=10, batch_size=1, callbacks=[callback],
                    verbose=0)

print(len(history.history['loss']))