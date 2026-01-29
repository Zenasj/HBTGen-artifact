# tf.random.uniform((B, 331, 331, 3), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense expects input shape (None, 331, 331, 3)
        # The first Dense layer is applied per spatial location (applied to last dimension only)
        # This matches the example from the issue.
        self.dense1 = tf.keras.layers.Dense(10, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(10, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)   # shape: (B, 331, 331, 10)
        x = self.flatten(x)       # shape: (B, 331*331*10=1095610)
        x = self.dense2(x)        # shape: (B, 10)
        return x

def cmetrics(y_true, y_pred):
    # Custom metric always returns zero scalar.
    # Matches the minimal metric from the issue.
    return tf.constant(0, dtype=tf.float32)

def my_model_function():
    # Return an instance of MyModel with compiled loss, optimizer, and metric
    model = MyModel()
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[cmetrics]
    )
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel:
    # Shape: (batch_size, 331, 331, 3),
    # dtype: float32 (standard for images)
    batch_size = 4  # A reasonably small batch size matching the example runs
    x = tf.random.uniform(shape=(batch_size, 331, 331, 3), dtype=tf.float32)
    return x

