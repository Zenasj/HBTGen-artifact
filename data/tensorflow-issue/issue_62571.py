import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, kernel_size=3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation="relu")
   ])

model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(10, kernel_size=3, activation="relu"),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, activation="relu")
 ])