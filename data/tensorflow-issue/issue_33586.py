# tf.random.normal((54, 4), dtype=tf.float32) and tf.random.normal((54,), dtype=tf.float32) ← Input shapes for boxes and scores/classes respectively

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, max_boxes=10, iou_threshold=0.5, score_threshold=0.1, soft_nms_sigma=0.0):
        super(MyModel, self).__init__()
        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms_sigma = soft_nms_sigma

    @tf.function(jit_compile=True)
    def call(self, inputs):
        """
        inputs: tuple of (scores, boxes, classes)
          scores: float32 tensor of shape (N,)
          boxes: float32 tensor of shape (N, 4)
          classes: float32 or int tensor of shape (N,)

        Returns:
          Tuple (scores_out, boxes_out, classes_out) after non max suppression
          Each with shape (<= max_boxes, ...)
        """
        scores, boxes, classes = inputs

        max_output_size = tf.constant(self.max_boxes, dtype=tf.int32)
        # Use tf.image.non_max_suppression which corresponds to NonMaxSuppressionV3 internally

        # We do NOT place the op explicitly on GPU because in TF 1.15 NonMaxSuppression ops
        # do not have GPU kernels, which leads to an InvalidArgumentError.
        # Instead, we rely on allow_soft_placement or default device placement to place
        # NMS on CPU automatically while rest of graph can run on GPU.

        nms_indices = tf.image.non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold
        )

        # Gather filtered boxes, scores, classes
        filtered_scores = tf.gather(scores, nms_indices)
        filtered_boxes = tf.gather(boxes, nms_indices)
        filtered_classes = tf.gather(classes, nms_indices)

        return filtered_scores, filtered_boxes, filtered_classes


def my_model_function():
    # Return an instance of MyModel with default parameters suitable for typical usage
    return MyModel(max_boxes=10, iou_threshold=0.5, score_threshold=0.1)


def GetInput():
    # Generate valid random input tensors matching expected shapes and dtypes
    
    # Using float32, as required for tf.image.non_max_suppression compatibility on GPU/CPU
    N = 54  # Number of boxes / scores
    
    # boxes: shape (N,4): each box is [ymin, xmin, ymax, xmax] format float32
    # Generate plausible box coordinates with ymin < ymax and xmin < xmax in [0, 1]
    ymin = tf.random.uniform((N,), minval=0, maxval=0.5, dtype=tf.float32)
    xmin = tf.random.uniform((N,), minval=0, maxval=0.5, dtype=tf.float32)
    ymax = tf.random.uniform((N,), minval=0.5, maxval=1.0, dtype=tf.float32)
    xmax = tf.random.uniform((N,), minval=0.5, maxval=1.0, dtype=tf.float32)
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

    # scores: random float32 confidence scores
    scores = tf.random.normal((N,), mean=1.0, stddev=4.0, dtype=tf.float32, seed=1)

    # classes: random int32 class indices from 0 to 9 for example
    classes = tf.random.uniform((N,), minval=0, maxval=10, dtype=tf.int32, seed=1)

    return scores, boxes, classes

