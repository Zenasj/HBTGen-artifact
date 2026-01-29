# tf.random.uniform((1, 10, 1, 4), dtype=tf.float32), tf.random.uniform((1, 10, 5), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We encapsulate the CombinedNonMaxSuppression logic in a layer here
        # to keep consistency with original code
        self.nms_layer = NMSLayer()

    def call(self, inputs, **kwargs):
        # inputs is a tuple/list of two tensors: boxes and scores
        boxes, scores = inputs
        return self.nms_layer([boxes, scores])

class NMSLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        boxes, scores = inputs[0], inputs[1]
        # Use tf.image.combined_non_max_suppression with parameters as per original example
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
        # Cast classes to int32 to match original behavior
        return boxes, scores, tf.cast(classes, dtype=tf.int32), valid_detections

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of inputs matching the model input signature:
    # boxes: shape (1, 10, 1, 4), float32
    # scores: shape (1, 10, 5), float32
    # Use random uniform data as dummy input
    boxes = tf.random.uniform((1, 10, 1, 4), minval=0, maxval=1, dtype=tf.float32)
    scores = tf.random.uniform((1, 10, 5), minval=0, maxval=1, dtype=tf.float32)
    return (boxes, scores)

