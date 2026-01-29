# tf.random.uniform((N_SAMPLES, 10), dtype=tf.float32) ← inferred input shape based on issue code

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This is a custom EarlyStopping callback that ensures the best weights
    are restored at the end of training regardless of whether training was
    stopped early or not, addressing the reported TensorFlow issue.

    Since the original issue revolves around EarlyStopping callback, the actual
    'model' used in the example is a simple Sequential with Dense layers.
    Here, we encapsulate both the model and the "EarlyStopping with guaranteed
    best weights restore" as submodules inside a single class for demonstration.

    The forward pass simply delegates to the internal model.
    """

    def __init__(self, n_classes=5, n_samples=500):
        super().__init__()
        # Base model matching the example in the issue
        self.base_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(n_classes)
        ])

        # Initialize a custom early stopping callback with guaranteed weights restore
        self.early_stopping = ReturnBestEarlyStopping(
            monitor="val_accuracy",
            patience=10,
            verbose=1,
            restore_best_weights=True
        )

    def call(self, inputs, training=False):
        # Forward pass through the base model
        return self.base_model(inputs, training=training)

    def get_early_stopping_callback(self):
        # Provide early stopping callback instance for use during training
        return self.early_stopping


class ReturnBestEarlyStopping(tf.keras.callbacks.EarlyStopping):
    """
    Custom EarlyStopping callback subclass that ensures restoring best weights
    at the end of training, regardless of whether early stopping triggered or
    training completed normally.

    This fixes the TensorFlow 2.0.0 EarlyStopping behavior where weights are only
    restored if early stopping happened, but not if training naturally ends.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_weights = None

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        # Reset best weights at train start
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = self._get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            self.best = current
            # Save a copy of the weights from the best epoch
            self.best_weights = [w.numpy() for w in self.model.weights]
            self.stopped_epoch = 0  # Reset stopped epoch if best found

        elif self.stopped_epoch == 0:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: early stopping')

    def on_train_end(self, logs=None):
        # Always restore the best weights after training ends if requested
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            # Restore weights
            for v, best_w in zip(self.model.weights, self.best_weights):
                v.assign(best_w)


def my_model_function():
    """
    Returns an instance of MyModel.

    This includes the wrapped model and early stopping callback.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor matching the expected model input.
    From the issue example, input shape is (N_SAMPLES, 10) where N_SAMPLES=500.
    We can just generate a single batch with shape (1,10) as a valid input.
    """
    # Use float32 dtype as usual in TF models
    return tf.random.uniform((1, 10), dtype=tf.float32)

