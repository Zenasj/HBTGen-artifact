import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

sample = splitted_blocks[0][0][6]
size = 300
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def resize_img(image, label):
    return tf.image.resize(image, (size, size)), label


ds_train = ds_train.map(resize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(resize_img, num_parallel_calls=tf.data.AUTOTUNE)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(size, size, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
sample = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)
sample = cv.resize(sample, (size, size))

view_image(sample)
sample = np.invert(np.array([sample]))

prediction = model.predict(sample)
print(np.argmax(prediction))

for class_index, prob in enumerate(prediction[0]):
    print(f'Class {class_index}: Probability {prob}')

sample = np.invert(np.array([sample])).reshape((size, size, 1))
cv.imshow("test", sample)
cv.waitKey(0)