# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape for MNIST images reshaped with channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the CNN model similar to the one described in the issue:
        # Conv2D(32, 3, activation='relu', input_shape=(28,28,1))
        # MaxPooling2D()
        # Flatten()
        # Dense(64, activation='relu')
        # Dense(10) output logits
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Instantiate the model and compile with similar parameters used in the reported code
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

def GetInput():
    # Return a random input tensor matching the expected input shape: batch size arbitrary (e.g., 4), 28x28 grayscale
    # Data type: float32 (normalized pixel values expected)
    batch_size = 4
    input_tensor = tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)
    return input_tensor

