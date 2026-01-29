# tf.random.uniform((B, 2), dtype=tf.float32)  <-- Input shape inferred from example: batch size B, 2 features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

        # Following original design: two copies of dense layer chains with sizes [2, 1]
        # Stored in a list of lists that remain append-only to avoid checkpointing errors.
        self._optimizers = []
        self._denses = []
        # Instead of caching grouped variables in a list that can be modified after creation
        # causing ListWrapper replacement, always compute on demand.
        # This avoids the problematic mutable list state that breaks saving weights.

        for _ in range(2):
            self._optimizers.append(tf.keras.optimizers.Adam())
            # Each 'copy' has two Dense layers
            self._denses.append([
                tf.keras.layers.Dense(2),
                tf.keras.layers.Dense(1)
            ])

    @property
    def groupedVariables(self):
        """
        Return a list of trainable variables grouped by the same grouping as self._denses.

        WARNING: Must compute each time to avoid mutable list issues that break saving.
        """
        var_lists = []
        for denses in self._denses:
            vars_for_denses = []
            # Append variables for each dense in the inner list
            for d in denses:
                vars_for_denses.extend(d.trainable_variables)
            var_lists.append(vars_for_denses)
        return var_lists

    def call(self, x):
        # Forward pass returns a list of outputs (one per group of denses).
        outputs = []
        for denses in self._denses:
            y = x
            for d in denses:
                y = d(y)
            outputs.append(y)
        return outputs

    def update(self, x, t):
        # Compute loss and gradients with respect to grouped variables.
        losses = []
        with tf.GradientTape() as tape:
            preds = self(x)  # List of outputs per dense group
            # Compute mse loss per output vs target t. Broadcasting allowed.
            for y in preds:
                losses.append(tf.reduce_mean(tf.square(t - y)))
            total_loss = tf.reduce_sum(losses)  # sum losses for gradient computation

        grouped_vars = self.groupedVariables
        grads = tape.gradient(total_loss, grouped_vars)
        # Apply gradient updates for each optimizer on its grouped variables
        for g, v, opt in zip(grads, grouped_vars, self._optimizers):
            if g is not None:
                opt.apply_gradients(zip(g, v))

def my_model_function():
    return MyModel()

def GetInput():
    # Create a random floating tensor of shape (1, 2)
    # consistent with example and model dense input shape.
    return tf.random.uniform((1, 2), dtype=tf.float32)

