# tf.random.uniform((32, 5), dtype=tf.float32) ← inferred input shape from batch_size=32, n_features=5

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer with sigmoid activation as per original example
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")
        # Binary crossentropy loss function instance, from_logits=False since output is sigmoid activated
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        # Accuracy metric instance to compute accuracy for batch
        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    def call(self, inputs, training=False):
        # Forward pass through the Dense layer
        return self.dense(inputs)

    def test_on_batch(self, x, y):
        # Compute predictions
        y_pred = self(x, training=False)
        # Calculate loss value
        loss = self.loss_fn(y, y_pred)
        # Update accuracy metric state using predictions and labels
        self.accuracy_metric.reset_states()
        self.accuracy_metric.update_state(y, y_pred)
        acc = self.accuracy_metric.result()
        return loss, acc

    def evaluate(self, x, y, batch_size=None, verbose=0):
        # Mimics Keras evaluate behavior on given dataset x,y, potentially batch-wise
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.batch(32)  # default batch size if none specified

        total_loss = 0.0
        total_acc = 0.0
        count = 0

        self.accuracy_metric.reset_states()

        for batch_x, batch_y in dataset:
            y_pred = self(batch_x, training=False)
            loss = self.loss_fn(batch_y, y_pred)
            self.accuracy_metric.update_state(batch_y, y_pred)
            total_loss += loss.numpy()
            count += 1
        acc = self.accuracy_metric.result().numpy()

        avg_loss = total_loss / count if count > 0 else 0.0

        # Return avg loss, accuracy similar to Keras.Model.evaluate()
        return avg_loss, acc

def my_model_function():
    model = MyModel()
    # Normally model weights are randomly initialized by default
    # No pretrained weights specified in the issue
    return model

def GetInput():
    # Generate a random input tensor of shape (batch_size=32, n_features=5) matching the model input
    # dtype is float32 as typical for TF models
    return tf.random.uniform((32, 5), dtype=tf.float32)

