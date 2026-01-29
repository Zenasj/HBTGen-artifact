# tf.random.uniform((B, H, W), dtype=tf.int32) ← Inferred input shape is arbitrary; here we assume (batch_size, height, width) for segmentation labels

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model encapsulates the custom MeanIoU metric with ignore_label functionality
    as a metric submodule.
    
    Given y_true (ground truth labels) and y_pred (logits or probabilities),
    it performs the argmax on y_pred per pixel, then updates the MeanIoU metric
    ignoring a specified label if provided.
    """
    def __init__(self, num_classes=3, ignore_label=None):
        super(MyModel, self).__init__()
        # Store number of classes and ignore_label (e.g. might be 255 or similar)
        self.num_classes = num_classes
        self.ignore_label = ignore_label

        # Instantiate base MeanIoU metric to track intersection-over-union
        self.metric = tf.keras.metrics.MeanIoU(num_classes=num_classes, name="mean_iou_ignore_label")

    def call(self, inputs):
        """
        Forward method expects inputs to be a tuple/list of (y_true, y_pred_logits).
        
        Returns the scalar metric value after updating with current inputs.
        """
        y_true, y_pred_logits = inputs
        # Convert logits / probabilities to predicted labels via argmax
        y_pred = tf.argmax(y_pred_logits, axis=-1, output_type=tf.int32)

        # Compute sample_weight to ignore the specified label, if any
        sample_weight = None
        if self.ignore_label is not None:
            # sample_weight is 0 where label == ignore_label, 1 elsewhere
            sample_weight = tf.cast(tf.not_equal(y_true, self.ignore_label), dtype=tf.float32)

        # Update underlying MeanIoU metric state
        self.metric.update_state(y_true, y_pred, sample_weight=sample_weight)
        # Return current MeanIoU value
        result = self.metric.result()
        return result

def my_model_function():
    """
    Returns an instance of MyModel with default configuration.
    You can customize num_classes and ignore_label as needed.
    """
    return MyModel(num_classes=3, ignore_label=None)

def GetInput():
    """
    Returns a valid input tuple (y_true, y_pred_logits) for MyModel.
    
    y_true: integer tensor (labels) shape (batch_size, height, width), dtype tf.int32, values in [0, num_classes-1]
    y_pred_logits: float tensor, shape (batch_size, height, width, num_classes), logits or probabilities
    
    This input matches assumptions inside MyModel.call.
    """
    batch_size = 4
    height = 15
    width = 20
    num_classes = 3

    # Random integer labels for y_true with values in class range
    y_true = tf.random.uniform(
        shape=(batch_size, height, width),
        minval=0, maxval=num_classes,
        dtype=tf.int32,
    )

    # Random logits for y_pred
    y_pred_logits = tf.random.uniform(
        shape=(batch_size, height, width, num_classes),
        minval=-1.0, maxval=1.0,
        dtype=tf.float32,
    )

    return (y_true, y_pred_logits)

