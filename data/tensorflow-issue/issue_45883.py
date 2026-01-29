# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape from example inp = tf.keras.Input((1,))

import tensorflow as tf

class Counter(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = self.add_weight('count', dtype=tf.int64, initializer='zeros')

    def update_state(self, *args, **kwargs):
        self.count.assign_add(1)

    def result(self):
        return self.count

    def reset_state(self):
        # Explicitly define reset_state to clear count
        self.count.assign(0)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A single Dense layer with 2 units and softmax activation (as per original example)
        self.dense = tf.keras.layers.Dense(2, activation='softmax')
        # Custom Counter metric instance
        self.counter = Counter(name="counter")

    def call(self, inputs, training=False):
        # Forward pass through dense layer
        return self.dense(inputs)

    def test_on_batch(self, x, y, reset_metrics=True, return_dict=False):
        # This method mimics tf.keras.Model.test_on_batch but clarifies and fixes the reset_metrics semantics
        # If reset_metrics True => reset BEFORE computing metrics (to get per-batch values)
        # If False => accumulate metrics statefully

        # If reset_metrics, reset metric state before updating it for this batch
        if reset_metrics:
            self.counter.reset_state()

        # Forward pass (no training)
        y_pred = self.call(x, training=False)

        # Compute loss (SparseCategoricalCrossentropy as in example)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_value = loss_fn(y, y_pred)

        # Update metric state with this batch's data
        # The custom Counter metric ignores inputs and just counts updates
        self.counter.update_state(y, y_pred)

        # Retrieve metric results
        metric_result = self.counter.result()

        if return_dict:
            return {"loss": loss_value, "counter": metric_result}
        else:
            return [loss_value, metric_result]

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching input shape (batch_size=1, input_dim=1) as example
    batch_size = 1
    input_shape = (batch_size, 1)
    x = tf.random.uniform(input_shape, dtype=tf.float32)
    # Target labels: integer class labels compatible with SparseCategoricalCrossentropy
    y = tf.zeros((batch_size,), dtype=tf.int64)
    return x, y

