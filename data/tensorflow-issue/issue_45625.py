# tf.random.uniform((B, 180, 180, 3), dtype=tf.float32) ← Input shape inferred from the dataset and model input_shape

import tensorflow as tf
from tensorflow.keras import layers, Sequential

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Rebuild the Sequential model from the issue with the same architecture
        self.model = Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(180, 180, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(5)  # num_classes=5 from the issue
        ])

    def call(self, inputs):
        return self.model(inputs)

def my_model_function():
    # Return an instance of MyModel.
    # The model is not trained here (weights uninitialized),
    # since original training happens outside.
    return MyModel()

def GetInput():
    # Return a random tensor input with shape (batch, 180, 180, 3), float32
    # Batch size can be arbitrary; choosing 1 for simplicity
    return tf.random.uniform((1, 180, 180, 3), dtype=tf.float32)

