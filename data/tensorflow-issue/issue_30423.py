from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

for zero_output_for_mask in [False, True]:
  #tf.keras.Sequential Model
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(16, 5, mask_zero=True),
      tf.keras.layers.GRU(5, return_sequences=True, zero_output_for_mask=zero_output_for_mask)
  ]
  )

  model.compile(
      optimizer='rmsprop',
      loss='mse'
      )

  np_x = np.ones((2, 5), dtype=np.float32)

  # masking timestamp 3,4 of sample index 1.
  np_x[1, 3:] = 0
  #print(np_x)

  # model's call()
  result_mask_call = model(np_x)

  print("---- zero_output_for_mask is {} ------".format(zero_output_for_mask))
  print(result_mask_call[1,:].numpy())