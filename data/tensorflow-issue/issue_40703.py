# tf.random.uniform((B, 10), dtype=tf.float32) ← inferred input shape from example x shape (10, 10)

import tensorflow as tf

class Count(tf.keras.metrics.Metric):
    def __init__(self, name=None, dtype=None, **kwargs):
        super(Count, self).__init__(name=name, dtype=dtype, **kwargs)
        # Use int64 for count to avoid overflow on large counts
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.int64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true flattened to get the first tensor (to handle nested structure)
        first_tensor = tf.nest.flatten(y_true)[0]
        batch_size = tf.shape(first_tensor)[0]
        self.count.assign_add(tf.cast(batch_size, tf.int64))

    def result(self):
        return tf.identity(self.count)

    def reset_states(self):
        self.count.assign(0)


class PrintInfo(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        # logs dict is empty in on_train_batch_begin due to regression in TF 2.2+
        # No batch size info is available here in logs as per issue discussion
        print(f"Start of batch {batch}; logs keys: {list(logs.keys()) if logs else 'None or empty'}")

    def on_train_batch_end(self, batch, logs=None):
        # logs contain metrics including Count metric 'counter'
        counter = logs.get('counter') if logs else None
        print(f"End of batch {batch}")
        print(f"Samples seen this epoch (from Count metric): {counter}")



class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple example model to match the example in the issue: one Dense layer outputting 1 value
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)


def my_model_function():
    model = MyModel()
    # We compile the model with SGD optimizer, MSE loss and the custom Count metric
    # This aligns with reproducible example provided in the issue
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[Count(name='counter')]
    )
    return model


def GetInput():
    # Returns a random tensor with shape matching the example input used in the issue
    # In the example, x shape is (10, 10), so batch size 10, feature dim 10
    # Using float32 default dtype
    return tf.random.uniform((10, 10), dtype=tf.float32)

