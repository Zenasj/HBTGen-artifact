# tf.random.uniform((3,), dtype=tf.float32) ← Input shape inferred from test y_true,y_pred shape (3,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Internally instantiate keras.metrics.Accuracy object
        # We will compare tf.keras.metrics.accuracy functional output vs Accuracy metric output
        self.acc_obj = tf.keras.metrics.Accuracy(name='my_acc')

    def call(self, inputs):
        """
        inputs: tuple or list of two 1-D tensors (y_true, y_pred)
        y_true expected: integer class labels (0 or 1)
        y_pred expected: float predictions or probabilities (matching shape)
        
        Returns a dict with:
        - "functional_accuracy": output of tf.keras.metrics.accuracy (functional API)
        - "metric_accuracy": output of tf.keras.metrics.Accuracy object update and result
        - "match": boolean scalar whether the two accuracies are equal
        """
        y_true, y_pred = inputs

        # The tf.keras.metrics.accuracy function computes mean of correct predictions.
        functional_acc = tf.keras.metrics.accuracy(y_true, y_pred)

        # Reset the state of the internal metric (for idempotence per call)
        self.acc_obj.reset_states()
        self.acc_obj.update_state(y_true, y_pred)
        metric_acc = self.acc_obj.result()

        # Compare the results for equality (note: float equality might be sensitive, 
        # so use tf.experimental.numpy.isclose or tf.math.abs diff < 1e-8)
        close = tf.math.abs(functional_acc - metric_acc) < 1e-8

        return {
            "functional_accuracy": functional_acc,
            "metric_accuracy": metric_acc,
            "match": close
        }

def my_model_function():
    # Return an initialized MyModel instance
    return MyModel()

def GetInput():
    # Generate inputs corresponding to the test case:
    # y_true = [1, 0, 1] (integer labels)
    # y_pred = [0.8, 0.1, 0.9] (floats, prob-like scores)
    y_true = tf.constant([1, 0, 1], dtype=tf.int32)
    y_pred = tf.constant([0.8, 0.1, 0.9], dtype=tf.float32)
    return (y_true, y_pred)

