from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

class TestBlock(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.shortcut = tf.keras.Sequential()
    self.dense = tf.keras.layers.Dense(5)
    self.relu = tf.keras.layers.ReLU()

  def call(self, input_vector):
    shortcut = self.shortcut(input_vector)
    dense = self.dense(input_vector)
    return self.relu(shortcut + dense)

if __name__ == "__main__":
  model = TestBlock()
  model.build((1,5))
  model.compile(loss="categorical_crossentropy")
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints",
    save_best_only=True,
    mode="auto",
  )
  train_data = tf.data.Dataset.from_tensor_slices((np.ones((20,5)), np.zeros((20,5)))).batch(1)
  model.fit(train_data,
    epochs=1,
    callbacks=[checkpoint_callback],
  )
  print("Saving ...")
  model.save("test_model")
  print("Loading ...")
  loaded_model = tf.keras.models.load_model("test_model")