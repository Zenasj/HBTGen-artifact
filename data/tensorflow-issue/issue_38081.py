from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import typing

import tensorflow as tf
from tensorflow import python as tfp

def save_serving_model(export_dir: str, model: tf.keras.Model,
                       classes: typing.List[str]):
    """Obtain a TensorFlow Serving function that can be used with
    `tf.saved_models.save`

    Args:
        export_dir: The directory in which to export the saved model.
        model: A classification model
        classes: The list of classification labels
    """

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serve(inputs):
        def map_fn(image):
            image = tf.io.decode_image(
                image['image'], channels=3, expand_animations=False)
            image = tf.cast(image, dtype=tf.float32)
            image = tf.compat.v1.image.resize_image_with_pad(
                image=image,
                target_height=model.input_shape[1],
                target_width=model.input_shape[2],
                align_corners=False,
                method=tf.image.ResizeMethod.BILINEAR)
            return image

        images = tf.io.parse_example(
            inputs,
            features={
                'image': tf.io.FixedLenFeature(shape=[], dtype=tf.string)
            },
            example_names=None,
            name=None)
        X = tf.map_fn(
            fn=map_fn, elems=images, back_prop=False, dtype=tf.float32)
        y = model.call(X)
        labels = tf.constant([classes])
        return {
            'scores': y,
            'classes': tf.repeat(
                labels, repeats=tf.shape(y)[0], axis=0, name=None)
        }

    # This is a super-ugly hack, but we have to do it because for
    # the API does not allow us to specify a method name. See:
    # https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/saved_model/save.py#L462
    try:
        PREDICT_METHOD_NAME = tfp.saved_model.signature_constants.PREDICT_METHOD_NAME
        tfp.saved_model.signature_constants.PREDICT_METHOD_NAME = tf.saved_model.CLASSIFY_METHOD_NAME  # pylint: disable=line-too-long
        tf.saved_model.save(
            model,
            export_dir,
            signatures={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                serve.get_concrete_function()
            })
    finally:
        tfp.saved_model.signature_constants.PREDICT_METHOD_NAME = PREDICT_METHOD_NAME

import tensorflow as tf

inputs = tf.keras.layers.Input((100, 100, 3))
x = tf.keras.layers.Conv2D(kernel_size=(2, 2), filters=3)(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Activation('softmax')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

save_serving_model(export_dir='export', model=model, classes=['class1', 'class2', 'class3'])

import requests
import glob

with open('../tests/images/image1.jpg', 'rb') as f:
    examples = [{
        'image': {
            'b64': base64.b64encode(f.read()).decode('utf-8')
        }
    }]
response = requests.post(
    'http://host.docker.internal:8501/v1/models/testmodel:classify',
    json={
        'examples': examples
    }
).json()
print(response['results'])