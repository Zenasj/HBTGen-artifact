from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(FEATURE_SIZE,)))
model.add(tf.keras.layers.Dense(1))
optim = tf.keras.optimizers.Adam()
model.compile(loss='mean_squared_error', optimizer=optim, metrics=['mae'])

x, y = load_features(train_files, scaler)
model.fit(x, y, epochs=num_epochs, callbacks=callbacks)

job_dir = '/path/to/job/'
version = '00000123'
export_path = os.path.join(job_dir, version)

tf.keras.models.save_model(
                model,
                export_path,
                overwrite=True,
                include_optimizer=True,
                save_format=None,
                options=None,
            )

updater = tf.compat.v1.saved_model.builder.MethodNameUpdater(export_path)
updater.replace_method_name(signature_key="bar", method_name="classify", tags="serve")
updater.save(export_path)

import tensorflow as tf 
import os 

FEATURE_SIZE = 3
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(FEATURE_SIZE,)))
model.add(tf.keras.layers.Dense(1))
optim = tf.keras.optimizers.Adam()
model.compile(loss='mean_squared_error', optimizer=optim, metrics=['mae'])

x = tf.ones((3, 3))
y = tf.zeros(3)

num_epochs = 3
model.fit(x, y, epochs=num_epochs)

job_dir = os.getcwd()
version = '1'
export_path = os.path.join(job_dir, version)

tf.keras.models.save_model(
                model,
                export_path,
                overwrite=True,
                include_optimizer=True,
                save_format=None,
                options=None,
            )

updater = tf.compat.v1.saved_model.builder.MethodNameUpdater(export_path)
updater.replace_method_name(signature_key="bar", method_name="classify", tags="serve")
updater.save(export_path)