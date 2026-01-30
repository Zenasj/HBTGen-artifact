import random
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tempfile
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Make Fake Data
x_train = np.random.rand(256, 784).astype("float32")
y_train = np.random.randint(low=0, high=10, size=256)

x_test = np.random.rand(64, 784).astype("float32")
y_test = np.random.randint(low=0, high=10, size=64)

# Build toy model 
inputs = keras.Input(shape=(x_train.shape[1]))
dense = layers.Dense(64, activation="relu")(inputs)
dense = layers.Dense(64, activation="relu")(dense)
outputs = layers.Dense(10)(dense)

model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

# Set Parameters
batch_size = 32 # divisible by number of examples in train (256)
location_save = tempfile.gettempdir() # Get location of temp directory to not store data permanently

# Align modelcheckpoint to end of epoch based on TF version. 
# Note only tf == 2.0 has been tried, as it was what is available on Conda (recommended install method)
if ((tf.__version__[:3] == '2.0') or
	(tf.__version__[:3] == '2.1')
	):
	save_freq = x_train.shape[0]
elif ((tf.__version__[:3] == '2.2') or
	  (tf.__version__[:3] == '2.3')
	  ):
	save_freq = x_train.shape[0] // batch_size

verbose = 2 # To easily see number of epochs and "warning" print statements from TF. 

# Create model checkpoint
model_checkpoint = ModelCheckpoint(filepath=os.path.join(location_save, 
													     'temp_mnist_weights.h5'),
                                   monitor='val_loss',
                                   verbose=verbose,
                                   save_best_only=True,
                                   save_weights_only=True,
                                   save_freq=save_freq
                                   )

# Fit model 
model.fit(x_train, 
		  y_train,
		  validation_data=(x_test, y_test),
		  batch_size=batch_size, 
		  epochs=2, 
		  callbacks=[model_checkpoint],
		  verbose=verbose)