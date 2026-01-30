from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


class NMSLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        boxes, scores = inputs[0], inputs[1]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size_per_class=8,
            max_total_size=8,
            iou_threshold=0.5,
            score_threshold=0.5,
            pad_per_class=False,
            clip_boxes=True,
            name=f'{self.name}/NMS_op'
        )

        return boxes, scores, tf.cast(classes, dtype=tf.int32), valid_detections


def representative_dataset_generator():
    for batch in range(10):
        yield [tf.zeros((1, 10, 1, 4)), tf.zeros((1, 10, 5))]


num_boxes = 10
num_classes = 5

boxes = tf.keras.Input((num_boxes, 1, 4), batch_size=1)
scores = tf.keras.Input((num_boxes, num_classes))

model = tf.keras.Model([boxes, scores], NMSLayer()([boxes, scores]))
# test that the model can run
model(next(representative_dataset_generator()))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.representative_dataset = representative_dataset_generator
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.allow_custom_ops = True
converter.experimental_new_converter = True