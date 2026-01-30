import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
import platform

print()
print(f"PLATFORM:\n---------\n{platform.platform()}")
print("\nTENSORFLOW:\n----------")
for a in tf.version.__all__:
    print(f"{a}: {getattr(tf.version, a)}")

print(f"\nNUMPY:\n-----\n{np.version.version}")

print(f"\nPYTHON:\n-------\n{sys.version}\n")

np.random.seed(0)
tf.random.set_seed(0)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation="linear")
])
model.compile(optimizer="sgd", loss="mse")

x = np.random.uniform(size=(1,1))
y = np.random.uniform(size=(1,))
model.fit(x, y, epochs=1)