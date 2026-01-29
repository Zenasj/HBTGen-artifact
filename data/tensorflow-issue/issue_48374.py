# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST (28x28 grayscale images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the CNN model as per example:
        # Input shape (28,28) → reshape (28,28,1)
        self.reshape = tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        self.conv2d = tf.keras.layers.Conv2D(32, 3, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10)  # logits for 10 classes

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.conv2d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate and compile model as in the example:
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model

def GetInput():
    # Return a batch of random inputs matching the input shape expected by MyModel:
    # The original data was normalized to [0,1] float32 values.
    # Batch size chosen as 64 to match 'global_batch_size' in example.
    batch_size = 64
    input_shape = (28, 28)  # single MNIST image shape

    # Generate random floats in [0,1), shape (batch_size, 28, 28)
    return tf.random.uniform((batch_size,) + input_shape, dtype=tf.float32)

