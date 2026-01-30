import tensorflow as tf
from tensorflow import keras

#Load dataset
train_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(224, 224),
  crop_to_aspect_ratio=True,
  shuffle=True,
  batch_size=BATCH_SIZE)

#Data augmentation
rescale = tf.keras.Sequential([
  layers.Rescaling(1./255)
])
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

def prepare(ds, shuffle=False, augment=False,batch_size=18):
  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (rescale(x), y), 
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets.
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)

# Augment the data
train_ds = prepare(train_ds,shuffle=True,augment=True,batch_size=BATCH_SIZE)