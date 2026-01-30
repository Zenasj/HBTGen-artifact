import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

IMG_SIZE = 128
NUM_BOXES = 100
CROP_SIZE = 28
NB_DATA_SAMPLES = 32
BATCH_SIZE = 1


class CropLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        images_, boxes_ = inputs
        box_indices = tf.reshape(
            tf.repeat(
                tf.expand_dims(tf.range(tf.shape(images_)[0], dtype=tf.int32), axis=-1),
                NUM_BOXES,
                axis=-1
            ),
            shape=(-1,)
        )
        cropped_images = tf.image.crop_and_resize(
            image=images_,
            boxes=tf.reshape(boxes_, (-1, 4)),
            box_indices=box_indices,
            crop_size=(CROP_SIZE, CROP_SIZE))
        return cropped_images


images = tf.random.normal(
    shape=(NB_DATA_SAMPLES, IMG_SIZE, IMG_SIZE, 3))
boxes = tf.random.uniform((NB_DATA_SAMPLES, NUM_BOXES, 4), maxval=1)

layer = CropLayer()
# Ensure that it is working, should be (NB_DATA_SAMPLES * NUM_BOXES, CROP_SIZE, CROP_SIZE, 3)
print('Should be (NB_DATA_SAMPLES * NUM_BOXES, CROP_SIZE, CROP_SIZE, 3), i.e. (3200, 28,28, 3):')
print(layer([images, boxes]).shape)

inputs_model = [
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), batch_size=BATCH_SIZE),
    tf.keras.Input(shape=(NUM_BOXES, 4), batch_size=BATCH_SIZE)
]

model = tf.keras.models.Model(inputs=inputs_model, outputs=layer(inputs_model))


def representative_dataset_generator():
    for image, bboxes in zip(images.numpy(), boxes.numpy()):
        image_ = tf.expand_dims(image, 0)
        bboxes_ = tf.expand_dims(bboxes, 0)
        yield [image_, bboxes_]

# Run the model on the representative dataset generator
for sample in representative_dataset_generator():
    model(sample).shape

# Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

print('Converting without representative_dataset')
quant_model = converter.convert()
with open('model.tflite', "wb") as f:
    f.write(quant_model)
print('Converted and saved')

print('Converting with representative_dataset')
converter.representative_dataset = representative_dataset_generator
quant_model = converter.convert()