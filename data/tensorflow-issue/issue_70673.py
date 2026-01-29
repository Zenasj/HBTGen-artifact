# tf.random.uniform((B, 512, 512, 3), dtype=tf.float32)  # B=batch size

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Reshape, Dropout
from tensorflow.keras.metrics import Metric
import numpy as np

# Assumptions and inferred constants based on the issue description:
# - bbox output shape: (6,4)  -> bbox_dim1=6, bbox_dim2=4
# - class_label output shape: (6,3) -> label_dim1=6, label_dim2=3
# - number of classes for IoU metric: 3 (consistent with shape and categories)
# Note: The bbox regression output uses sigmoid activation consistent with normalized coords.
# The class_label head uses softmax activation and categorical crossentropy loss.
# We will encapsulate the original model in MyModel.
# Also implement a custom IoU metric wrapper that handles one-hot label input and bounding boxes,
# to illustrate the likely comparison of metrics and the place where ScatterNd error might originate.

bbox_dim1 = 6
bbox_dim2 = 4
label_dim1 = 6
label_dim2 = 3
num_classes = 3

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
        # Freeze conv layers
        for layer in self.base_model.layers:
            layer.trainable = False
        
        self.flatten = Flatten()

        # Bounding box head
        self.bbox_dense1 = Dense(128, activation="relu")
        self.bbox_dense2 = Dense(64, activation="relu")
        self.bbox_dense3 = Dense(32, activation="relu")
        self.bbox_output = Dense(bbox_dim1 * bbox_dim2, activation="sigmoid")
        self.bbox_reshape = Reshape((bbox_dim1, bbox_dim2))

        # Class label head
        self.class_dense1 = Dense(512, activation="relu")
        self.class_dropout1 = Dropout(0.5)
        self.class_dense2 = Dense(512, activation="relu")
        self.class_dropout2 = Dropout(0.5)
        self.class_output = Dense(label_dim1 * label_dim2, activation="softmax")
        self.class_reshape = Reshape((label_dim1, label_dim2))

        # We instantiate an IoU metric for demonstration, but as in the original code,
        # it can expect integer labels, so the mismatch may cause the ScatterNd error.
        self.iou_metric = tf.keras.metrics.IoU(num_classes=num_classes, target_class_ids=[0,1])

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=False)
        x = self.flatten(x)

        bbox = self.bbox_dense1(x)
        bbox = self.bbox_dense2(bbox)
        bbox = self.bbox_dense3(bbox)
        bbox = self.bbox_output(bbox)
        bbox = self.bbox_reshape(bbox)

        class_label = self.class_dense1(x)
        if training:
            class_label = self.class_dropout1(class_label, training=training)
        class_label = self.class_dense2(class_label)
        if training:
            class_label = self.class_dropout2(class_label, training=training)
        class_label = self.class_output(class_label)
        class_label = self.class_reshape(class_label)

        # Return dict of outputs to match original model signature
        return {"bounding_box": bbox, "class_label": class_label}

    @tf.function(jit_compile=True)
    def compute_iou_metric(self, y_true, y_pred):
        # This function emulates where ScatterNd error could appear if y_true or y_pred have incompatible shapes
        # The original IoU metric expects sparse integer labels (class indices), but we provide one-hot labels,
        # so this mismatch might cause that scatter error in graph mode.

        # Convert one-hot y_true to class indices for IoU metric
        true_class_indices = tf.argmax(y_true, axis=-1)
        pred_class_indices = tf.argmax(y_pred, axis=-1)

        self.iou_metric.update_state(true_class_indices, pred_class_indices)
        return self.iou_metric.result()

def my_model_function():
    # Returns a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random input tensor compatible with MyModel:
    # Shape: (batch_size, 512, 512, 3), dtype float32
    batch_size = 10  # typical batch size used in the issue dataset

    # Random float image input 0-1
    inputs = tf.random.uniform((batch_size, 512, 512, 3), dtype=tf.float32)
    return inputs

