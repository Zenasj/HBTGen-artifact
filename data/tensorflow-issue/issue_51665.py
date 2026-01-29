# tf.random.uniform((128, 1), dtype=tf.float32) ← inferred input shape (BATCH_SIZE=2^7=128, STATE_DIM=1)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 2 ** 7
NUM_ACTION = 11
STATE_DIM = 1

def _huber_loss(y_true, y_pred, max_grad=1.):
    """Huber loss element-wise."""
    a = tf.abs(y_true - y_pred)
    less_than_max = 0.5 * tf.square(a)
    greater_than_max = max_grad * (a - 0.5 * max_grad)
    return tf.where(a <= max_grad, x=less_than_max, y=greater_than_max)

def mean_huber_loss(y_true, y_pred):
    """Mean Huber loss over the batch."""
    return tf.reduce_mean(_huber_loss(y_true, y_pred))

class MyModel(tf.keras.Model):
    def __init__(self):
        # We do not pass inputs and outputs upfront since we want a cleaner subclass
        super(MyModel, self).__init__()
        # Define a simple Dense layer matching original code outputs
        self.dense = keras.layers.Dense(NUM_ACTION)
        # Metrics:
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.abs_metric = keras.metrics.MeanTensor(name="abs")  # tracks mean absolute error tensor
        
        # Loss function: mean huber loss as defined above
        self.criterion = mean_huber_loss

    def call(self, inputs, training=False):
        # Forward pass through Dense layer
        return self.dense(inputs)

    @tf.function
    def train_step(self, data):
        states, targets = data

        with tf.GradientTape() as tape:
            logits = self(states, training=True)
            # Using compiled_loss corresponds to loss passed to model.compile
            # If user wants to avoid saving issues, can switch to criterion directly.
            loss = self.compiled_loss(targets, logits)
            # Alternative which works for saving: 
            # loss = self.criterion(targets, logits)

        # Compute gradients and apply optimizer
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.loss_tracker.update_state(loss)
        # abs_metric tracks mean absolute error per sample, resulting in shape [batch_size]
        self.abs_metric.update_state(tf.math.reduce_mean(tf.math.abs(targets - logits), axis=-1))

        return {"loss": self.loss_tracker.result(), "abs": self.abs_metric.result()}

    @property
    def metrics(self):
        # Must list metrics so that they get reset automatically each epoch
        return [self.loss_tracker, self.abs_metric]


def my_model_function():
    # Create an instance of MyModel and compile it with optimizer and loss
    model = MyModel()
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.1,
        first_decay_steps=1000
    )
    # Here pass mean_huber_loss to compiled_loss to mimic original example.
    # Note: in original issue, using compiled_loss caused save errors,
    # but it is the usage pattern shown.
    model.compile(optimizer=Adam(lr_schedule), loss=mean_huber_loss)
    return model

def GetInput():
    # Return a random input that has the expected model input shape (batch_size, STATE_DIM)
    # Use tf.float32 dtype for compatibility
    return tf.random.uniform((BATCH_SIZE, STATE_DIM), dtype=tf.float32)

