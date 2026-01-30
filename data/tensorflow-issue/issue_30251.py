from tensorflow import keras
from tensorflow.keras import layers

tf.keras.experimental.export_saved_model(
    model, file_path,
    serving_only=True,
    input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)]
)

import tensorflow as tf
dataset1 = tf.data.Dataset.from_tensor_slices([[10, 11], [12, 13], [14, 15], [16, 17]])
dataset2 = tf.data.Dataset.from_tensor_slices([[21, 22], [23, 24], [25, 26], [27, 28]])
sample_dataset = tf.data.experimental.sample_from_datasets(
  [dataset1, dataset2], weights=[0.5, 0.5], seed=43)

my_strategy = tf.distribute.MirroredStrategy(["GPU:0", "CPU:0"])
with my_strategy.scope():
  @tf.function
  def distribute_train_epoch(dataset):
    for x in dataset:
      my_strategy.run(print, args=(x,))
  # Create a tf.keras model.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(1, input_shape=[10]))
  model.summary()

    # Save the tf.keras model in the SavedModel format.
  path = 'simple_keras_model'
  tf.compat.v1.keras.experimental.export_saved_model(model, path)
  dist_dataset = my_strategy.experimental_distribute_dataset(sample_dataset)

for _ in range(2):
  print("------------------")
  distribute_train_epoch(dist_dataset)