# tf.random.uniform((B, 10), dtype=tf.float32) ← input is a batch of feature vectors with shape (batch_size, 10)

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Using DenseFeatures layer with numeric_column keyed to 'x'
        self.features = tf.keras.layers.DenseFeatures([
            tf.feature_column.numeric_column('x', shape=(10,))
        ])
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        # inputs is expected to be a dict with key 'x' among potentially others (e.g., 'z')
        x = self.features(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def my_model_function():
    # Instantiate and compile the model consistent with the original example
    model = MyModel()
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05)
    )
    return model


def GetInput():
    # Generate a batch of inputs compatible with the model.
    # The model expects inputs as a dict with key 'x' containing float32 tensors of shape (B, 10).
    # The 'z' key is unused in the model but commonly included in the original input_fn,
    # so include it here to match expected input dict structure.
    batch_size = 32  # typical batch size used in example

    # Generate random feature data for 'x'
    x = tf.random.uniform((batch_size, 10), dtype=tf.float32)
    # Create a zero tensor matching x for key 'z' (not used in model but part of input dict)
    z = tf.zeros_like(x)

    return {'x': x, 'z': z}

