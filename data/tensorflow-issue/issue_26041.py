import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self._layer1 = tf.keras.layers.Dense(20, activation='relu')
        self._layer2 = tf.keras.layers.Dense(10)

    def call(self, x, training):
        x = self._layer1(x)
        x = self._layer2(x)
        return x


def create_dataset():
    def process(features):
        image, label = features['image'], features['label']
        return tf.reshape(image, [-1]) / np.float32(255.0), label

    data_builder = tfds.builder('mnist')
    dataset = data_builder.as_dataset(split=tfds.Split.TRAIN)
    dataset = (
        dataset.map(process)
        .batch(32)
        .repeat(1)
    )

    return dataset

  
avg_loss = tf.metrics.Mean()

  
@tf.function
def train(model, optimizer):
    dataset = create_dataset()
    step = 0
    for images, labels in dataset:
        step += 1
        with tf.GradientTape() as tape:
            logits = model(images, True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            loss = tf.math.reduce_mean(loss)
            
        avg_loss.update_state(loss)
        
        grads = tape.gradient(
            loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables))
        
        if tf.equal(step % 20, 0):
            tf.print(avg_loss.result())
            avg_loss.reset_states()
            

NUM_EPOCHS = 2
model = MyModel()
optimizer = tf.keras.optimizers.Adam()
for _ in range(NUM_EPOCHS):
    train(model, optimizer)