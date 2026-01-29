# tf.random.uniform((1, 320, 320, 3), dtype=tf.float32) ← Inferred input shape from TF Lite model input details (typical for SSD MobileNet)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    """
    A placeholder/fused model class for SSD MobileNet V2 FPNLite 320x320 TF2 detection model,
    reflecting the context of the issue where TF Lite conversion fails on tf.Size op.
    This class simulates a preprocessing + model inference + postprocessing pipeline as a single Keras model.

    Since the problem is about model conversion and TFLite inputs/outputs shape mismatch,
    this class illustrates:
    - Expected input shape: [1, 320, 320, 3]
    - Outputs four tensors similar to TFLite SSD detection models: boxes, classes, scores, count

    The forward pass returns a tuple of these outputs, as numpy-like tensors.
    """

    def __init__(self):
        super().__init__()
        # We simulate a backbone + detection head with simple layers to yield expected output shapes.
        # This is a dummy structure to satisfy input/output shapes and signatures for conversion.

        self.backbone = tf.keras.applications.MobileNetV2(
            input_shape=(320,320,3),
            include_top=False,
            weights=None,
            pooling='avg'  # Global average pooling for simplicity
        )
        # Detection head layers:
        self.dense_boxes = tf.keras.layers.Dense(4 * 10)    # 10 boxes output, 4 coords each
        self.dense_scores = tf.keras.layers.Dense(10)       # 10 scores
        self.dense_classes = tf.keras.layers.Dense(10)      # 10 classes (float output indices)
        self.count = tf.Variable(10, trainable=False, dtype=tf.int32)  # Always 10 detections

    def call(self, inputs, training=False):
        x = self.backbone(inputs, training=training)  # shape [B, feature_dim]
        # Predict boxes
        boxes = self.dense_boxes(x)  # [B, 40]
        # Reshape to [B, 10, 4]
        boxes = tf.reshape(boxes, (-1, 10, 4))

        # Normalize boxes to be between 0 and 1 (sigmoid)
        boxes = tf.sigmoid(boxes)

        # Predict classes - softmax for 1 class + background (simulate)
        classes = self.dense_classes(x)  # [B,10]
        classes = tf.sigmoid(classes)

        # Predict scores 
        scores = self.dense_scores(x)
        scores = tf.sigmoid(scores)

        # Count tensor as constant tensor broadcasted per batch
        count_tensor = tf.fill([tf.shape(inputs)[0]], self.count)  # [B]

        return boxes, classes, scores, count_tensor


def my_model_function():
    # Return instance of MyModel.
    # In a real scenario, weights should be loaded here.
    return MyModel()


def GetInput():
    # Create a random input tensor shaped to what model expects.
    # Here batch size 1, height 320, width 320, 3 channels, dtype float32.
    return tf.random.uniform((1, 320, 320, 3), dtype=tf.float32)

