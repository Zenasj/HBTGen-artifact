import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# file demo.py
import sys
import numpy as np
import tensorflow as tf

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
graph_context = tf.Graph().as_default()
strategy_context = strategy.scope()

with graph_context:
    with strategy_context:
        ip = tf.keras.layers.Input([2])
        h = tf.keras.layers.Dense(10, activation='relu', input_dim=2)(ip)
        out = tf.keras.layers.Dense(2, activation='softmax')(h)
        model = tf.keras.models.Model(inputs=[ip], outputs=[out])

        model.compile(optimizer='Adam',
                loss='categorical_crossentropy',
                metrics=[tf.keras.metrics.AUC(num_thresholds=100)])

    x = np.random.randn(100, 2)
    y = (x[:, 0] * x[:, 1]) > 0
    model.fit(x, tf.keras.utils.to_categorical(y), epochs=1)