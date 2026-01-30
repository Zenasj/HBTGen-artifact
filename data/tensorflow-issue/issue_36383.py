from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.mixed_precision import experimental as mixed_precision


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def preprocess(ex):
    image = tf.cast(tf.expand_dims(ex['image'], axis=0), tf.float32) / 255.
    image = tf.image.resize(image, [224, 224])

    return image, ex['label']


data = tfds.load(name='imagenette/full-size', split="train", data_dir="/home/glorre/tensorflow_datasets",
                 shuffle_files=True).repeat()
data = data.map(preprocess).batch(5)

image = keras.Input(shape=[1, 224, 224, 3], name="seq")
features = keras.layers.TimeDistributed(
    keras.applications.ResNet50(include_top=False, weights=None, pooling='avg'))(image)

features = keras.layers.Lambda(lambda x: tf.squeeze(x, axis=[1]))(features)

logits = keras.layers.Dense(10)(features)
logits = keras.layers.Activation('linear', dtype='float32')(logits)

model = keras.Model(inputs=image, outputs=logits)

optimizer_1 = keras.optimizers.Adam(0.01)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer_1, loss=loss)

model.summary()

model.fit(data, epochs=10, steps_per_epoch=10)