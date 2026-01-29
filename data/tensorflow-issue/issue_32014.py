# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from Fashion MNIST grayscale images

import tensorflow as tf

class Metric_HIGH_COST(tf.keras.metrics.Metric):
    def __init__(self, name='high_cost_ce', dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        # Track the sum of sigmoid cross entropy losses
        self.ce = self.add_weight(name='ce', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Assign directly to the variable the sum of sigmoid cross entropy losses on batch
        ce_val = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        self.ce.assign(ce_val)

    def result(self):
        # Return the tracked loss value
        return self.ce

    def reset_states(self):
        self.ce.assign(0.0)


class Metric_LOW_COST(tf.keras.metrics.Metric):
    def __init__(self, ce_variable, name='low_cost_ce', dtype=None):
        """
        Wrap a variable that is updated from loss function for metric reporting.
        Assumes ce_variable is a tf.Variable that stores the loss value updated by loss.
        """
        super().__init__(name=name, dtype=dtype)
        # Store reference to the variable updated inside the loss
        self.ce = ce_variable

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Do nothing; value is updated inside loss function
        pass

    def result(self):
        # Return current value of variable
        return self.ce.read_value()

    def reset_states(self):
        # Reset the variable to zero
        self.ce.assign(0.0)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the model architecture like original sequential model in the issue
        self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=1, input_shape=[28, 28, 1])
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.lrelu1 = tf.keras.layers.LeakyReLU()

        self.conv2 = tf.keras.layers.Conv2D(64, 3, strides=1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.lrelu2 = tf.keras.layers.LeakyReLU()

        self.conv3 = tf.keras.layers.Conv2D(128, 3, strides=1)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.lrelu3 = tf.keras.layers.LeakyReLU()

        # Note the original model had two LeakyReLU layers here before flatten
        self.extra_lrelu = tf.keras.layers.LeakyReLU()

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.lrelu4 = tf.keras.layers.LeakyReLU()
        self.dense_out = tf.keras.layers.Dense(10)

        # This variable will be updated inside the loss call for Metric_LOW_COST
        self.ce_variable = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='ce_variable')

        # Instantiate metrics inside the model for direct access if desired
        self.metric_high_cost = Metric_HIGH_COST(name='high_cost_ce')
        self.metric_low_cost = Metric_LOW_COST(self.ce_variable, name='low_cost_ce')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.lrelu2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.lrelu3(x)

        x = self.extra_lrelu(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn4(x, training=training)
        x = self.lrelu4(x)

        logits = self.dense_out(x)
        return logits

# Custom loss class implementing the behavior described in the issue
class Myloss(tf.keras.losses.Loss):
    def __init__(self, model_ce_variable, name='myloss'):
        super().__init__(name=name)
        # Reference the shared variable from the model to update in loss
        self.ce = model_ce_variable

    def call(self, y_true, y_pred):
        # Compute sigmoid cross entropy summed over all elements
        ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

        # Assign the loss value to the track variable in loss, to share with Metric_LOW_COST
        # The expression "ce_loss + 0 * self.ce.assign(ce_loss)" forces tf.function to run assign op,
        # but does not affect returned loss value.
        loss_with_update = ce_loss + 0 * self.ce.assign(ce_loss)
        return loss_with_update


def my_model_function():
    model = MyModel()
    # Create instance of loss with model variable reference
    loss_obj = Myloss(model.ce_variable)

    # Instantiate metrics based on the model's metric objects
    # Return the model; user can compile with loss_obj and metrics as needed.
    # Example:
    # model.compile(optimizer='adam', loss=loss_obj,
    #               metrics=[model.metric_high_cost, model.metric_low_cost])
    # Because the metric_low_cost reads from the variable updated in loss, it reflects loss state.
    return model, loss_obj, model.metric_high_cost, model.metric_low_cost


def GetInput():
    # Return a batch of inputs corresponding to Fashion MNIST images:
    # Assuming batch size 8 for example, dtype float32
    batch_size = 8
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

