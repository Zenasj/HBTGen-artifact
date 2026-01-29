# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← typical CIFAR-10 input shape used in the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple CIFAR-10-like CNN model as mentioned in the discussion
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2,2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2,2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        # Output logits for 10 classes (CIFAR-10 style)
        self.dense_out = tf.keras.layers.Dense(10)

        # For demonstration, we will instantiate two metrics to compare:
        # - tf.keras.metrics.SparseCategoricalAccuracy (correct usage for multi-class)
        # - tf.keras.metrics.Accuracy (which expects binary or matching shapes on preds and labels)
        # Note: These show the core issue described in the reports.
        self.metric_sparse_cat_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')
        self.metric_accuracy = tf.keras.metrics.Accuracy(name='accuracy')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense_out(x)
        return logits

    def compute_metrics(self, y_true, y_pred_logits):
        """
        Update both metrics. Note y_pred_logits are raw logits for 10 classes.
        SparseCategoricalAccuracy expects labels of shape (batch,) and logits or probs.
        Accuracy expects both inputs be the same shape and matching dtype.

        So, with y_true shape (batch, 1) or (batch,), and y_pred_logits (batch, 10),
        Accuracy metric's update_state will raise an error due to incompatible shapes.

        To simulate, we convert logits to predicted class indices via argmax for Accuracy.
        """
        # Predicted classes for Accuracy metric
        y_pred_classes = tf.argmax(y_pred_logits, axis=-1, output_type=y_true.dtype)

        # Update SparseCategoricalAccuracy directly with logits and y_true
        self.metric_sparse_cat_acc.update_state(y_true, y_pred_logits)

        # Update Accuracy with y_true and predicted classes (to avoid shape mismatch)
        # This mimics the scenario difference highlighted in the issue.
        self.metric_accuracy.update_state(y_true, y_pred_classes)

    def result_metrics(self):
        # Return the metric results as a dictionary
        return {
            'sparse_categorical_accuracy': self.metric_sparse_cat_acc.result(),
            'accuracy': self.metric_accuracy.result()
        }

    def reset_metrics(self):
        self.metric_sparse_cat_acc.reset_state()
        self.metric_accuracy.reset_state()

def my_model_function():
    """
    Returns an instance of MyModel.
    This model encapsulates the difference between using 'Accuracy' metric
    (tf.keras.metrics.Accuracy) and the more suitable
    'SparseCategoricalAccuracy' metric for multi-class classification.
    """
    return MyModel()

def GetInput():
    """
    Returns a tuple (input_tensor, labels) compatible with MyModel.

    Input is a random float tensor mimicking CIFAR-10 images: shape (batch, 32, 32, 3).
    Labels are integers representing class indices, shape (batch,) or (batch,1).

    This matches the format needed for sparse categorical metrics and avoids shape mismatches.
    """
    batch_size = 4
    input_tensor = tf.random.uniform((batch_size, 32, 32, 3), minval=0, maxval=1, dtype=tf.float32)
    labels = tf.random.uniform((batch_size,), minval=0, maxval=10, dtype=tf.int32)
    return input_tensor, labels

