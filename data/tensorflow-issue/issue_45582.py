import random
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow import keras


class WeightChecker:
    """Automated health checks for training Keras models."""
    def __init__(self, model):
        self.initial_model = model
        self.var_names = [var.name for var in model.trainable_variables]
        self.prev_weights = model.get_weights()

    def check_epoch(self, model):
        """Checks to run at the end of an epoch"""
        self.check_untrained_params(model)

    def check_untrained_params(self, model):
        """Compare self.model.trainable_variables to self.prev_weights"""
        passed = True
        curr_weights = model.get_weights()
        for curr_var, prev_var, var_name in zip(curr_weights, self.prev_weights, self.var_names):
            eq = np.equal(curr_var, prev_var).all()
            if eq:
                passed = False
                print(f"\nWarning: Variable {var_name} was not updated with training. "
                      f"Confirm that this layer is correctly "
                      f"connected to the computation graph.")
        self.prev_weights = [w.copy() for w in curr_weights]
        return passed


class WeightCheckerCallback(keras.callbacks.Callback):
    """Check model initialization and run training checks.
    """
    def __init__(self):
        super().__init__()
        self.weight_check = None

    def setup_weight_checker(
            self,
            model: keras.Model = None):
        """Initialize the callback with an input_batch and targets."""
        self.weight_check = WeightChecker(model)

    def on_train_begin(self, logs=None):
        if self.weight_check is None:
            raise ValueError("setup_weight_checker() must be called to use WeightCheckerCallback.")

    def on_epoch_end(self, epoch, logs=None):
        self.weight_check.check_epoch(self.model)


def data_gen():
    """Generate random data for training."""
    data = (np.random.random((audio_len, 1)).astype(np.float32),
            np.random.random((audio_len, 1)).astype(np.float32))
    while True:
        yield data


batch_size = 64
audio_len = 16000
steps_per_epoch = 10
dataset = tf.data.Dataset.from_generator(data_gen,
                                         (tf.float32, tf.float32),
                                         (tf.TensorShape((audio_len, 1)), tf.TensorShape((audio_len, 1))))
dataset = dataset.batch(batch_size)

map_model = tf.keras.Sequential([tf.keras.layers.Conv1D(
        64, 3, padding='same'
    ),
    tf.keras.layers.Conv1D(
        1, 3, padding='same'
    )])
map_model(np.random.random((batch_size, audio_len, 1)))

aux_model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
aux_model.trainable = False
aux_model(np.random.random((batch_size, 1)))


class MainModel(tf.keras.Model):
    """Main Model."""
    def __init__(self, map_model, aux_model):
        super().__init__()
        self.feature_dim = 128
        self.aux_model = aux_model
        self.map_model = map_model
        self.global_step = 0

    def call(self, inputs, training=True):
        output = self.map_model(inputs)
        return output

    def train_step(self, data):
        mixed_audio = data[0]
        clean_audio = data[1]

        with tf.GradientTape() as tape:
            decoded_audio = self.map_model(mixed_audio)
            total_loss = tf.reduce_mean(tf.abs(clean_audio - decoded_audio))

        grads = tape.gradient(total_loss, self.trainable_variables)
        [tf.print(f'Gradient std for {tv.name}: '
                  f'{np.std(g.numpy())}')
         for (g, tv) in zip(grads, self.trainable_variables)
         if 'conv1d/' in tv.name]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        losses = {
            'loss': total_loss,
        }
        # tf.summary.scalar('loss', total_loss, step=self.global_step)
        self.global_step += 1
        return losses


epochs = 5

model = MainModel(map_model, aux_model)
model.compile(
    loss='mae',
    optimizer='adam',
    run_eagerly=True
)

weight_checker = WeightCheckerCallback()
weight_checker.setup_weight_checker(model)

model.fit(
    dataset,
    epochs=epochs,
    callbacks=[weight_checker],
    steps_per_epoch=steps_per_epoch
)