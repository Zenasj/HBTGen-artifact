# tf.random.uniform((100, 8), dtype=tf.float32) ← Inferred input shape from example usage in the issue (100 samples, 8 features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple feedforward model as per the example in issue:
        self.dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')

        # Metrics as per discussed issue:
        # - The bug arises when using capitalized "Accuracy" string metric vs. lowercase 'accuracy'.
        # - Here, we instantiate both metrics explicitly to show the difference.
        self.accuracy_metric_correct = tf.keras.metrics.MeanMetricWrapper(
            tf.keras.metrics.BinaryAccuracy(), name='accuracy_wrapped'
        )
        self.accuracy_metric_incorrect = tf.keras.metrics.Accuracy(name='accuracy_raw')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)

        # Update both metrics
        # `accuracy_metric_correct` expects y_true and y_pred, so assume y_true is passed externally.
        # But since call normally only processes inputs, returning just predictions here.
        return output

    def train_step(self, data):
        # Unpack data: inputs and labels
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients and optimize
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics:
        # The MeanMetricWrapper wraps a BinaryAccuracy metric and gives correct accuracy
        self.accuracy_metric_correct.update_state(y, y_pred)
        # The raw Accuracy metric just compares argmax or exact values - often inappropriate here, will give zero
        # For binary outputs, threshold predictions at 0.5 for update_state of Accuracy
        y_pred_labels = tf.cast(y_pred > 0.5, tf.int32)
        y_int = tf.cast(y, tf.int32)
        self.accuracy_metric_incorrect.update_state(y_int, y_pred_labels)

        # Update compiled loss and other metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Prepare a dictionary of metrics for logs including both "correct" and "incorrect" accuracy to expose the issue
        logs = {m.name: m.result() for m in self.metrics}
        logs[self.accuracy_metric_correct.name] = self.accuracy_metric_correct.result()
        logs[self.accuracy_metric_incorrect.name] = self.accuracy_metric_incorrect.result()

        return logs

    def reset_metrics(self):
        super().reset_metrics()
        self.accuracy_metric_correct.reset_states()
        self.accuracy_metric_incorrect.reset_states()

def my_model_function():
    # Return an instance of MyModel with standard compile config similar to example:
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        # Metrics as blank because we handle metrics explicitly in train_step
        metrics=[]
    )
    return model

def GetInput():
    # Return a random input tensor matching shape (batch_size=100, features=8)
    # Also return matching labels as required for training and metrics calculation
    # Labels are binary targets (0 or 1)
    inputs = tf.random.uniform((100, 8), dtype=tf.float32)
    labels = tf.random.uniform((100,), maxval=2, dtype=tf.int32)  # binary labels 0 or 1
    return inputs, tf.cast(labels, tf.float32)

