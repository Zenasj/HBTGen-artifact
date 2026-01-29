# tf.random.uniform((B, 1), dtype=tf.float32)  # Input shape inferred from example: (batch_size, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single dense layer with 1 output, kernel initialized to ones, bias to zeros, input shape=1 feature
        self.dense = tf.keras.layers.Dense(1,
                                           kernel_initializer='ones',
                                           bias_initializer='zeros')

    def call(self, inputs, training=False):
        return self.dense(inputs)

    def test_step(self, data):
        """
        Custom test_step method to correctly compute weighted mean batch loss over an epoch.

        The default test_step averages batch losses equally (unweighted),
        which is incorrect when the last batch has fewer samples.
        Here we accumulate the sum of batch losses weighted by batch size,
        and keep track of total samples seen to compute a correct weighted mean.
        """
        x, y = data
        y_pred = self(x, training=False)
        # Compute per-sample loss (returns shape [batch_size, ...]), reduction = 'none'
        loss_fn = self.compiled_loss._losses[0] if self.compiled_loss._losses else tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        # fallback if the above is empty, to cover usage
        # Note: Normally model.compiled_loss is a LossesContainer with losses registered at compile time.

        # Compute loss per example, so we can sum and weight by actual batch size
        per_sample_loss = loss_fn(y, y_pred)  # shape (batch_size,)
        # Sum of losses in this batch
        batch_loss_sum = tf.reduce_sum(per_sample_loss)
        batch_size = tf.shape(y)[0]

        # Accumulate weighted loss sum and count of samples in loss metric
        # We use self.compiled_loss._loss_metric which tracks the accumulated loss
        # and override its update_state to add weighted sum of batch losses

        # Note: Because compiled_loss is internal, we handle carefully to avoid side effects.

        # Update the loss metric by sum of losses, weighted by 1 (since sum, not mean)
        self.compiled_loss._loss_metric.update_state(batch_loss_sum, sample_weight=1.0)
        # Accumulate the total number of seen samples
        if not hasattr(self, '_total_samples'):
            self._total_samples = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._total_samples.assign_add(tf.cast(batch_size, tf.int64))

        # Compute weighted mean loss over all samples seen so far
        weighted_mean_loss = self.compiled_loss._loss_metric.result() / tf.cast(self._total_samples, tf.float32)

        # Update metrics (for example loss and any other metrics)
        self.compiled_metrics.update_state(y, y_pred)
        # Report weighted mean loss as loss metric
        return {**{m.name: m.result() for m in self.metrics}, 'loss': weighted_mean_loss}

def my_model_function():
    # Return an instance of MyModel compiled with optimizer and loss
    model = MyModel()
    # Compile with SGD optimizer and Mean Squared Error loss
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

def GetInput():
    # Return a random input tensor shaped (batch_size, 1) with float32 values in [0, 10)
    # batch_size chosen arbitrarily to 4 for demonstration
    batch_size = 4
    return tf.random.uniform((batch_size, 1), minval=0, maxval=10, dtype=tf.float32)

