import os

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = models.Sequential([layers.Dense(1, activation='relu')])
optimizer = optimizers.SGD()

# Is this line needed in graph mode?
# model.build((None, 1))


@tf.function
def update(batch):
    with tf.GradientTape() as tape:
        output = model(batch)
    grads = tape.gradient(output, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


if __name__ == "__main__":

    batch = tf.zeros((1, 1), dtype=tf.float32)
    update(batch)