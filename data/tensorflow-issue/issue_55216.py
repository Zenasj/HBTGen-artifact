# tf.random.uniform((batch_size, 10), dtype=tf.float32) ← Input shape inferred from example code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        # Add a Dense layer as in the example: inputs (shape=(10,)) → outputs (shape=(10,))
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        # Forward pass using dense layer
        return self.dense(inputs)

    def compute_loss(self, x, y, y_pred, sample_weight=None):
        # Custom loss: Mean squared error + any added losses
        tf.print("compute_loss is called:")  # Print to verify if called
        loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y))
        loss += tf.add_n(self.losses) if self.losses else 0.0
        self.loss_tracker.update_state(loss)
        return loss

    def reset_metrics(self):
        self.loss_tracker.reset_states()

    @property
    def metrics(self):
        # Return metrics tracked by the model for Keras API
        return [self.loss_tracker]

def my_model_function():
    # Create an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected input shape (batch, 10)
    # Assume batch size 1 to match example Dataset batching
    return tf.random.uniform((1, 10), dtype=tf.float32)

