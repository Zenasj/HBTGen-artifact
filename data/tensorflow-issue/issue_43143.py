# tf.random.uniform((100, 10), dtype=tf.float32) ← Input shape inferred from example X shape (100 samples, 10 features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple two-output binary classification model, mimicking the example in the issue
        self.dense_hidden = tf.keras.layers.Dense(5, activation='relu')
        self.output_one = tf.keras.layers.Dense(1, activation='sigmoid', name='one')
        self.output_two = tf.keras.layers.Dense(1, activation='sigmoid', name='two')

        # Two separate BinaryCrossentropy loss instances for each head
        self.loss_one = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        self.loss_two = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, inputs, training=False):
        x = self.dense_hidden(inputs)
        out1 = self.output_one(x)
        out2 = self.output_two(x)
        return out1, out2

    def compute_loss(self, y_true, y_pred):
        # y_true and y_pred are expected as tuples/lists of (y_true_1, y_true_2) and (y_pred_1, y_pred_2)
        y_true_1, y_true_2 = y_true
        y_pred_1, y_pred_2 = y_pred

        # Compute per-sample loss for each output head (shape: [batch_size])
        loss_1 = self.loss_one(y_true_1, y_pred_1)
        loss_2 = self.loss_two(y_true_2, y_pred_2)

        # Compute mean per-head losses (scalar)
        mean_loss_1 = tf.reduce_mean(loss_1)
        mean_loss_2 = tf.reduce_mean(loss_2)

        # Total loss is sum of both (as typical in multi-output models)
        total_loss = mean_loss_1 + mean_loss_2

        return total_loss, mean_loss_1, mean_loss_2

    def compare_losses(self, y_true, y_pred):
        # A helper method to demonstrate comparison between:
        # 1) The combined loss over concatenated outputs
        # 2) The sum of losses computed separately on two heads
        #
        # This is to illustrate the discrepancy described in the issue.
        #
        # Concatenate predictions and labels as in the manual numpy logs:
        # pred tuples: (batch_size, 1), concatenate to (batch_size, 2)
        pred_concat = tf.concat([y_pred[0], y_pred[1]], axis=1)
        true_concat = tf.concat([y_true[0], y_true[1]], axis=1)

        # Compute binary crossentropy losses with TensorFlow builtin
        # This yields the average loss per sample over both outputs combined
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        combined_loss = bce(true_concat, pred_concat)

        # Compute losses separately and sum
        total_loss, loss_one, loss_two = self.compute_loss(y_true, y_pred)

        # Return both for comparison (note: combined_loss won't exactly equal total_loss)
        return combined_loss, total_loss, loss_one, loss_two

def my_model_function():
    # Return an instance of MyModel as defined above
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape expected: (batch_size=100, features=10), float32
    # Batch size corresponds to the minimal example in the issue
    return tf.random.uniform((100, 10), dtype=tf.float32)

