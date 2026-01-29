# tf.random.uniform((BATCH_SIZE, 28*28), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple MLP model for MNIST flattened (28*28 input)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2, seed=1)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

        # Two possible metric configurations to illustrate the issue:
        # 1. Using string 'accuracy' metric (which uses SparseCategoricalAccuracy internally)
        self.metric_str_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # 2. Using tf.metrics.Accuracy() -- causes shape mismatch if used directly with sparse labels
        self.metric_raw_accuracy = tf.metrics.Accuracy()

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

    def compute_metrics_comparison(self, y_true, y_pred):
        """
        Computes and returns metric values from both metrics (string-based SparseCategoricalAccuracy and raw Accuracy).
        This will illustrate the typical difference and the shape mismatch issue real kernels face.
        """
        # Note: y_pred is probability distribution for 10 classes
        # SparseCategoricalAccuracy expects (batch_size,) int labels matching argmax of predictions
        self.metric_str_accuracy.update_state(y_true, y_pred)

        # For raw Accuracy metric, the inputs need to be same shape after argmax for y_pred
        # We simulate how raw Accuracy expects labels to be given:
        # It compares predictions and labels element-wise
        # So first convert y_pred to predicted label indices:
        y_pred_labels = tf.argmax(y_pred, axis=-1)

        # This metric expects y_true and y_pred_labels same shape
        # For sparse labels shape (batch_size,), this is OK
        self.metric_raw_accuracy.update_state(y_true, y_pred_labels)

        return (self.metric_str_accuracy.result(), self.metric_raw_accuracy.result())

def my_model_function():
    return MyModel()

def GetInput():
    BATCH_SIZE = 32
    # MNIST flattened input shape (batch_size, 28*28), float32 normalized in [0,1]
    x = tf.random.uniform((BATCH_SIZE, 28*28), dtype=tf.float32)
    # Sparse labels: int32, values in [0-9], shape (batch_size,)
    y = tf.random.uniform((BATCH_SIZE,), maxval=10, dtype=tf.int32)
    return x, y

