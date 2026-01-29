# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST example with grayscale images sized 28x28

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # CNN layers as per original example
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Build and compile the model exactly as in the issue example:
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Create a random tensor mimicking a batch of grayscale 28x28 images.
    # Batch size chosen arbitrarily to 64, matching example batch size.
    batch_size = 64
    input_tensor = tf.random.uniform((batch_size, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)
    return input_tensor

# Notes / explanation:
# - The original issue describes a problem with multi-worker training when datasets are not repeated.
# - The dataset was MNIST images scaled to (0.,1.) and batch size 64.
# - The model is a small CNN: Conv2D(32,3) -> MaxPool -> Flatten -> Dense(64 relu) -> Dense(10 softmax).
# - The compiled model uses sparse categorical crossentropy and SGD with lr=0.001.
# - Input shape is (28, 28, 1), grayscale MNIST images.
# - The `GetInput()` function returns a batch of random inputs with shape (64, 28, 28, 1) and float32 dtype.
#   This matches expected input to the model (batch dimension + input dims).
# - No training loop or dataset creation is included since that pertains to distribution strategy usage in the issue,
#   but model and input replicate core components relevant to that.

