# tf.random.uniform((B, 32), dtype=tf.float32) ← Input shape inferred from data shape (1000, 32)

import tensorflow as tf

class ReservoirHistogram(tf.metrics.Metric):
    def __init__(self,
                 name='histogram',
                 reservoir_size=300,
                 reservoir_shape=(10,),
                 dtype=tf.float32,
                 **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.reservoir = self.add_weight(
            name='reservoir',
            shape=(reservoir_size,) + reservoir_shape,
            initializer='zeros',
            dtype=self.dtype)
        self.current_index = self.add_weight(
            name='current_index', shape=(), dtype=tf.int32,
            initializer=tf.constant_initializer(0))

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_pred)[0]
        reservoir_size = tf.cast(tf.shape(self.reservoir)[0], tf.int32)
        r_index = self.current_index % reservoir_size
        batch_part = tf.minimum(batch_size, reservoir_size - r_index)
        # Generate uniform random values for reservoir sampling mask
        batch_random = tf.random.uniform(shape=(batch_part,), maxval=1.)
        self.current_index.assign_add(batch_part)
        # Probability of accepting new sample into reservoir
        reservoir_prob = tf.cond(
            self.current_index < reservoir_size,
            lambda: 1.0,
            lambda: tf.cast(reservoir_size / tf.cast(self.current_index, tf.float32), tf.float32)
        )
        batch_mask = batch_random <= reservoir_prob
        # Expand mask dimensions to match reservoir feature shape for weighting
        mask_shape = tf.concat([[batch_part], tf.shape(self.reservoir)[1:]], axis=0)
        batch_mask = tf.tile(tf.expand_dims(tf.cast(batch_mask, tf.float32), axis=1), multiples=mask_shape[1:])
        batch_mask = tf.reshape(batch_mask, mask_shape)
        old_values = self.reservoir[r_index: r_index + batch_part]
        batch_values = y_pred[:batch_part]
        # Transform values to have a zero-centered distribution clipped between -0.1 and 0.1
        batch_values = tf.clip_by_value(
            tf.where(batch_values > 0.5, batch_values - 1, batch_values),
            clip_value_min=-0.1, clip_value_max=0.1)
        new_values = old_values * (1 - batch_mask) + batch_values * batch_mask
        # Reconstruct reservoir tensor by replacing updated slots
        new_reservoir = tf.concat([
            self.reservoir[:r_index],
            new_values,
            self.reservoir[r_index + batch_part:]
        ], axis=0)
        self.reservoir.assign(new_reservoir)

    def result(self):
        used_size = tf.minimum(self.current_index, tf.shape(self.reservoir)[0])
        # Return only the used part of the reservoir buffer
        return self.reservoir[:used_size]

    def reset_states(self):
        self.reservoir.assign(tf.zeros(shape=tf.shape(self.reservoir)))
        self.current_index.assign(0)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple sequential style model matching issue example
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')
        # Instantiate metrics:
        self.cat_acc = tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
        self.histogram_metric = ReservoirHistogram()

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        y_pred = self.output_layer(x)
        return y_pred

    def compute_metrics(self, y_true, y_pred):
        # Update the metrics as would happen during training loop
        self.cat_acc.update_state(y_true, y_pred)
        self.histogram_metric.update_state(y_true, y_pred)
        # Return dict of metric results
        return {
            'categorical_accuracy': self.cat_acc.result(),
            'histogram': self.histogram_metric.result()
        }

def my_model_function():
    # Return an instance of MyModel, all weights initialized randomly by default
    return MyModel()

def GetInput():
    # Return a random tensor input matching shape (batch_size, 32)
    # We pick batch 32 here to match dataset batch size from the example
    return tf.random.uniform((32, 32), dtype=tf.float32)

