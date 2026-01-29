# tf.random.uniform((B, 1, 3), dtype=tf.float32)

import tensorflow as tf
import tensorflow.keras as k

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.num_classes = 6  # activities in the training set
        self.num_features = 3  # sensor (x,y,z)
        self.batch_size = 32

        # Mapping int labels to string activity names using StaticHashTable
        self.mapping = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.range(self.num_classes, dtype=tf.int32),
                values=tf.constant(
                    [
                        "Walking",
                        "Jogging",
                        "Upstairs",
                        "Downstairs",
                        "Sitting",
                        "Standing",
                    ]
                ),
            ),
            default_value="Unknown",
        )

        # Model architecture: LSTM layer with stateful=True (state maintained across batches)
        # and a Dense layer to output logits for the 6 activity classes
        self._model = k.Sequential(
            [
                k.layers.Input(
                    shape=(1, self.num_features),
                    batch_size=self.batch_size,
                ),
                k.layers.LSTM(64, stateful=True),
                k.layers.Dense(self.num_classes),
            ]
        )

        # Variables for training state
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._last_tracked_activity = tf.Variable(-1, dtype=tf.int32, trainable=False)

        # Optimizer and loss for training
        self._optimizer = k.optimizers.SGD(learning_rate=1e-4)
        self._loss = k.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 1, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ]
    )
    def learn(self, sensor_data, labels):
        # Ensure all labels in the batch are the same activity (assertion)
        tf.debugging.assert_equal(labels, tf.zeros_like(labels) + labels[0])

        # Reset LSTM states if activity label changes from the last batch
        if tf.not_equal(self._last_tracked_activity, labels[0]):
            tf.print(
                "Resetting states. Was: ",
                self._last_tracked_activity,
                " is ",
                labels[0],
            )
            self._last_tracked_activity.assign(labels[0])
            self._model.reset_states()

        self._global_step.assign_add(1)

        with tf.GradientTape() as tape:
            logits = self._model(sensor_data)
            loss = self._loss(labels, logits)
            tf.print(self._global_step, ": loss: ", loss)

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        return loss

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1, 3), dtype=tf.float32)])
    def predict(self, sensor_data):
        logits = self._model(sensor_data)
        predicted = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        # Print the human-readable activity labels for predicted classes
        tf.print("Predicted activities:", self.mapping.lookup(predicted))

        return predicted

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 1, 3), dtype=tf.float32)
    ])
    def call(self, sensor_data):
        # Forward call maps to predict for simpler usage of the model instance
        return self.predict(sensor_data)


def my_model_function():
    # Instantiate and return the model
    return MyModel()


def GetInput():
    # Return a random tensor input matching the model's expected shape
    # Batch size matches model batch_size, steps=1, features=3
    model = my_model_function()
    batch_size = model.batch_size
    return tf.random.uniform((batch_size, 1, 3), dtype=tf.float32)

