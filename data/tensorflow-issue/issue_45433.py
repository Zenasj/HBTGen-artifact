# tf.random.uniform((3, 3), dtype=tf.float32), tf.random.uniform((3, 10), dtype=tf.float32) ← input shape for (inputs, targets)

import tensorflow as tf

class LogisticEndpoint(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = tf.keras.metrics.BinaryAccuracy()

    def call(self, logits, targets, sample_weights=None):
        # Compute training-time loss value and add it via self.add_loss()
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy metric
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # Return inference-time prediction tensor (softmax of logits)
        return tf.nn.softmax(logits)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(10)
        self.endpoint = LogisticEndpoint(name="predictions")

    def call(self, inputs):
        # Inputs is expected as a tuple (inputs_tensor, targets_tensor)
        # logits = dense(inputs)
        inputs_tensor, targets_tensor = inputs
        logits = self.dense(inputs_tensor)
        # Pass logits and targets to endpoint layer
        predictions = self.endpoint(logits, targets_tensor)
        return predictions


def my_model_function():
    # Create an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of (inputs, targets) matching what MyModel expects
    inputs = tf.random.uniform((3, 3), dtype=tf.float32)
    targets = tf.random.uniform((3, 10), dtype=tf.float32)
    return (inputs, targets)

