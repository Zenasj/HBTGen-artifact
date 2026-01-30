from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

# Create a simple model
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
model.compile(optimizer='adam', loss='mse')
model.save('模型/')  # '模型' is Chinese for 'model'

# Now, try to load the saved model
loaded_model = tf.keras.models.load_model('模型/')