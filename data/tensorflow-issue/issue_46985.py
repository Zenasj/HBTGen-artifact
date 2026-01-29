# tf.random.uniform((B, ...), dtype=tf.float32) ← Input shape is assumed to be (batch_size, ...) with float32 dtype

import tensorflow as tf
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Internal variables to store concatenated y_true and y_pred across batches/devices
        # Using tf.Variable with validate_shape=False to allow growing variable shape
        self.y_true = tf.Variable(initial_value=[0.], dtype=tf.float32,
                                  shape=tf.TensorShape(None),
                                  trainable=False,
                                  synchronization=tf.VariableSynchronization.ON_READ,
                                  aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                  validate_shape=False,
                                  name='y_true')
        self.y_pred = tf.Variable(initial_value=[0.], dtype=tf.float32,
                                  shape=tf.TensorShape(None),
                                  trainable=False,
                                  synchronization=tf.VariableSynchronization.ON_READ,
                                  aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                  validate_shape=False,
                                  name='y_pred')
        self.was_called = self.add_weight(name='was_called',
                                          initializer='zeros',
                                          dtype=tf.uint8,
                                          trainable=False,
                                          synchronization=tf.VariableSynchronization.ON_READ,
                                          aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    
    def update_state(self, y_true, y_pred):
        # Flatten inputs and cast to float32 (compatible with internal variables)
        y_true = K.flatten(K.cast(y_true, tf.float32))
        y_pred = K.flatten(K.cast(y_pred, tf.float32))
        # Concatenate new values to stored variables
        # Skip the initial dummy [0.] entry when returning results
        assign_true = self.y_true.assign(tf.concat([self.y_true, y_true], axis=0))
        assign_pred = self.y_pred.assign(tf.concat([self.y_pred, y_pred], axis=0))
        assign_called = self.was_called.assign(1)
        return tf.group(assign_true, assign_pred, assign_called)

    def result(self):
        # If update_state was called, compute custom metric on accumulated values ignoring first dummy element
        def compute_metric():
            # Placeholder custom metric implementation; user should replace this with actual metric function
            y_true_vals = self.y_true[1:]
            y_pred_vals = self.y_pred[1:]
            # For demonstration, use mean absolute error as example
            return tf.reduce_mean(tf.abs(y_true_vals - y_pred_vals))
        
        return tf.cond(self.was_called >= 1,
                       true_fn=compute_metric,
                       false_fn=lambda: 0.0)
    
    def reset_states(self):
        # Reset stored variables to initial dummy value and called flag to zero
        self.y_true.assign(tf.constant([0.], dtype=tf.float32))
        self.y_pred.assign(tf.constant([0.], dtype=tf.float32))
        self.was_called.assign(0)

    def call(self, inputs):
        # For demonstration, the model just outputs the current result of the metric.
        # In real use case, model should implement forward logic.
        return self.result()


def my_model_function():
    # Return an instance of MyModel; no special initialization or weights needed.
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input for the model.
    # Since the metric accepts y_true and y_pred, we provide a tuple of (y_true, y_pred)
    # Assume batch size = 5, scalar predictions/labels
    batch_size = 5
    y_true = tf.random.uniform((batch_size,), dtype=tf.float32)
    y_pred = tf.random.uniform((batch_size,), dtype=tf.float32)
    # The model expects calls to update_state(y_true, y_pred), so we return tuple here
    # Users of the model should call update_state explicitly, but for compatibility with model(input)
    # the call method returns the current metric
    return (y_true, y_pred)

