# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32), tf.random.uniform((B, None, 10), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Image branch
        self.conv2d = tf.keras.layers.Conv2D(3, 3)
        self.gmp2d = tf.keras.layers.GlobalMaxPooling2D()
        # Timeseries branch
        self.conv1d = tf.keras.layers.Conv1D(3, 3)
        self.gmp1d = tf.keras.layers.GlobalMaxPooling1D()
        # Concatenate dense layers
        self.concat = tf.keras.layers.Concatenate()
        self.score_dense = tf.keras.layers.Dense(1, name='score_output')
        self.class_dense = tf.keras.layers.Dense(5, activation='softmax', name='class_output')

        # Metrics to track individual accuracies
        self.score_accuracy = tf.keras.metrics.CategoricalAccuracy(name='score_categorical_accuracy')
        self.class_accuracy = tf.keras.metrics.CategoricalAccuracy(name='class_categorical_accuracy')

    def call(self, inputs, training=False):
        # Unpack inputs tuple (image, timeseries)
        image_input, ts_input = inputs

        # Image branch forward
        x1 = self.conv2d(image_input)
        x1 = self.gmp2d(x1)

        # Timeseries branch forward
        x2 = self.conv1d(ts_input)
        x2 = self.gmp1d(x2)

        # Concatenate
        x = self.concat([x1, x2])

        # Outputs
        score_output = self.score_dense(x)
        class_output = self.class_dense(x)

        return score_output, class_output

    def reset_metrics(self):
        self.score_accuracy.reset_states()
        self.class_accuracy.reset_states()

    def update_metrics(self, y_true, y_pred):
        # y_true and y_pred are tuples for multi-output: (score_true, class_true), (score_pred, class_pred)
        score_true, class_true = y_true
        score_pred, class_pred = y_pred

        # The original code used categorical accuracy on score_output which is shape (B,1)
        # but typically CategoricalAccuracy expects one-hot for multi-class. Here we keep as is.
        # We update metrics tracking categorical accuracy individually.
        self.score_accuracy.update_state(score_true, score_pred)
        self.class_accuracy.update_state(class_true, class_pred)

    def compute_weighted_categorical_accuracy(self):
        # Compute weighted accuracy = weighted sum of per-output categorical accuracies
        # Using the example loss_weights from original code: score_output weight = 2.0, class_output weight = 1.0
        weighted_acc = (2.0 * self.score_accuracy.result() + 1.0 * self.class_accuracy.result()) / 3.0
        return weighted_acc


def my_model_function():
    # Create a compiled Keras model with losses and metrics, matching the original example
    model = MyModel()

    # Wrap the Keras model into a functional one to support compile, losses and metrics

    # Define Inputs
    image_input = tf.keras.Input(shape=(32, 32, 3), name='img_input')
    timeseries_input = tf.keras.Input(shape=(None, 10), name='ts_input')

    # Get outputs by forwarding inputs through MyModel instance
    score_output, class_output = model.call((image_input, timeseries_input))

    # Create a tf.keras.Model for compilation and training
    keras_model = tf.keras.Model(inputs=[image_input, timeseries_input],
                                 outputs=[score_output, class_output])

    # Compile with losses and built-in metrics as original code
    # NOTE: score_output used MeanSquaredError loss, class_output used CategoricalCrossentropy loss
    # Metrics: categorical accuracy defined per output
    keras_model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-3),
        loss={
            'score_output': tf.keras.losses.MeanSquaredError(),
            'class_output': tf.keras.losses.CategoricalCrossentropy(),
        },
        metrics={
            'score_output': [tf.keras.metrics.CategoricalAccuracy()],
            'class_output': [tf.keras.metrics.CategoricalAccuracy()]
        },
        loss_weights={'score_output': 2.0, 'class_output': 1.0}
    )

    # We can additionally attach our MyModel instance to keras_model for metric combination if needed
    keras_model.my_model = model  # attach submodule for metric tracking if needed

    return keras_model

def GetInput():
    # Generate inputs matching the model's expected input shapes:
    # image_input: (batch_size, 32, 32, 3), dtype float32
    # timeseries_input: (batch_size, 20, 10), dtype float32 (fixed 20 timesteps here for convenience)
    batch_size = 2  # small batch size for testing
    image_input = tf.random.uniform((batch_size, 32, 32, 3), dtype=tf.float32)
    timeseries_input = tf.random.uniform((batch_size, 20, 10), dtype=tf.float32)
    return [image_input, timeseries_input]

