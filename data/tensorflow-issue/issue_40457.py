from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow.keras.layers as kl

model = tf.keras.Sequential(kl.Dense(4))

optim = tf.keras.optimizers.Adam(
    learning_rate=0.001,
)

loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True,
)

class_weights = {
	0: 1., 1: 0.5, 2: 0.8, 3: 1.3,
}

xs = tf.zeros((16, 5, 5, 1))  # img
ys = tf.zeros((16, 5, 5, 4))  # one hot label

model.compile(optim, loss)

# Runs but fails silently
model.fit(xs, ys, class_weight=class_weights.values())
# Broken for n_classes> 1 (but is written as-documented)
model.fit(xs, ys, class_weight=class_weights)

print("Hello")