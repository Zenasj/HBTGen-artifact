# tf.random.uniform((10, 1), dtype=tf.float32) ← inferred from DummyGenerator output batch shape (10, 1)

import tensorflow as tf
import numpy as np

class BatchCounter(tf.keras.layers.Layer):
    def __init__(self, name="batch_counter", **kwargs):
        super(BatchCounter, self).__init__(name=name, **kwargs)
        self.stateful = True
        # Using a Variable instead of Keras backend variable for better compatibility
        self.batches = tf.Variable(0, dtype=tf.int32, trainable=False)

    def reset_states(self):
        self.batches.assign(0)

    def call(self, y_true, y_pred):
        current_batches = self.batches.value()
        # Increment the batch count by 1
        self.batches.assign_add(1)
        # Return the batch count + 1 as a float metric, to mimic original behavior
        # (added 1 is consistent with original code)
        return tf.cast(current_batches + 1, tf.float32)

class DummyGenerator(object):
    """ Dummy data generator yielding batches of shape (10, 1) with features of ones and zeros as labels """
    def run(self):
        while True:
            yield np.ones((10, 1), dtype=np.float32), np.zeros((10, 1), dtype=np.float32)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(1)
        self.batch_counter_metric = BatchCounter()

        # Create a keras metrics.Metric wrapper around BatchCounter to integrate with model.metrics
        # But since the original BatchCounter is a Layer returning a float, we'll just implement metric like behavior
        # during training step here for demonstration.

    def call(self, inputs, training=False):
        outputs = self.dense(inputs)
        return outputs

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update BatchCounter metric by calling with y_true and y_pred
        batch_count_metric = self.batch_counter_metric(y, y_pred)

        # Update other metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Prepare results dict for logs()
        results = {m.name: m.result() for m in self.metrics}
        # Include our batch counter metric result keyed by its name
        results[self.batch_counter_metric.name] = batch_count_metric

        return results

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        # Update loss and metrics
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        results = {m.name: m.result() for m in self.metrics}
        return results

def my_model_function():
    # Create an instance of MyModel, compile it similarly to the example:
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]  # Add any standard metric for demonstration
    )
    return model

def GetInput():
    # Return a random tensor input matching input shape expected by MyModel, based on DummyGenerator shape: (batch=10, 1)
    return tf.random.uniform((10, 1), dtype=tf.float32)

