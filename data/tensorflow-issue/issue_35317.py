from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

# In order to reproduce, just use whatever random JPEG you have handy here.
# It should be larger than my_crop in the x and y dimension.
my_jpeg = "/home/ben/my_jpeg.jpg"
my_crop = 224

# Generates a single crop for TensorFlow.
class DataGenerator(keras.utils.Sequence):
  def __init__(
      self,
      image_location,
      crop_size=224):
    self._image_location = image_location
    self._crop_size = crop_size

  # Just one single batch will be returned, of just one single image.
  def __len__(self):
    return 1

  # Generate one batch of data.
  def __getitem__(self, index):
    # Where the tensors will be stored.
    X = []
    y = [1]

    # Read it.
    image = tf.io.read_file(self._image_location)

    # Load it.
    image = tf.image.decode_jpeg(image, channels=3)

    assert image.shape[2] == 3  # MUST be RGB.
    height = image.shape[0]
    width = image.shape[1]

    # Just take a trivial crop of the image.
    # This is the offending line operation which hangs forever.
    image = image[0:self._crop_size, 0:self._crop_size, :]

    # This line is equivalent to above, and it also hangs with multiprocessing enabled.
    # image = tf.slice(image, [0, 0, 0], [self._crop_size, self._crop_size, 3])

    X.append(tf.dtypes.cast(image, tf.float32))

    # Tensors are not generally assignable, but we can create them from a number of existing ones.
    X = tf.stack(X)
    y = tf.stack(y)

    # Preprocess it.
    X /= 255.0  # Normalize to [0, 1] range.

    return X, y

generator = DataGenerator(my_jpeg, my_crop)

model = tf.keras.applications.ResNet50(input_shape=(my_crop, my_crop, 3))

model.compile(loss='mse')

# use_multiprocessing=False works.
# use_multiprocessing=True hangs.
model.fit_generator(generator, use_multiprocessing=True, workers=2)