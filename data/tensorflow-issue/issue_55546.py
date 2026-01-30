from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.layers import Dense, Dropout, Layer, Input
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import Model

import tensorflow as tf
import numpy as np

class PositionalEmbedding(Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)

        self.units = units

        self.projection = Dense(units, kernel_initializer=TruncatedNormal(stddev=0.02))
        self.dropout = Dropout(rate=dropout_rate)

    def build(self, input_shape):
        super(PositionalEmbedding, self).build(input_shape)

        print(input_shape, '\n')
        self.position = self.add_weight(
            name="position",
            shape=(1, input_shape[1], input_shape[2], self.units),
            initializer=TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs, training):
        x = self.projection(inputs)
        x += self.position

        return self.dropout(x, training=training)

class Transformer(Model):
    def __init__(
        self,
        embed_dim,
        dropout_rate,
        **kwargs
    ):
        super(Transformer, self).__init__(**kwargs)

        # Input
        self.pos_embs = PositionalEmbedding(embed_dim, dropout_rate)

    def compile(self, optimizer, loss):
        super(Transformer, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def call(self, inputs, training):
        inputs, targets = inputs
        
        return self.pos_embs(inputs, training=training)

    def train_step(self, inputs):
        inputs, targets = inputs

        print(inputs.shape)
        print(targets.shape, '\n')

        targets_inputs = targets[:, :-1]
        targets_real = targets[:, 1:]

        with tf.GradientTape() as tape:
            y_pred = self([inputs, targets_inputs], training=True)
            loss = self.loss(targets_real, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "loss": loss,
        }

def load_dataset(batch_size, window_size):
  x_all = np.ones((1000, 25, 81)) #np.load('./dataset/X_all.npy')
  y_all = np.ones((1000, 25, 1)) #np.load('./dataset/y_all.npy')

  inputs_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
      x_all, None, sequence_length=window_size, sequence_stride=(window_size // 2), batch_size=batch_size)
  targets_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
      y_all, None, sequence_length=window_size, sequence_stride=(window_size // 2), batch_size=batch_size)

  return inputs_dataset, targets_dataset



# load dataset
inputs_dataset, targets_dataset = load_dataset(
      batch_size=64,
      window_size=7,
)
dataset = tf.data.Dataset.zip((inputs_dataset, targets_dataset))

for batch in dataset:
    inputs, targets = batch
    print(inputs.shape)
    print(targets.shape, '\n')
    break

sample_transformer = Transformer(
    embed_dim=256, dropout_rate=0.1,
)

sample_transformer.compile(
      loss=tf.keras.losses.mean_squared_error,
      optimizer=tf.keras.optimizers.Adam(),
)

# Train model
sample_transformer.fit(
      dataset,
      epochs=10
)