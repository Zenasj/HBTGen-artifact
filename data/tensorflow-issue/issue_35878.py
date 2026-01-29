# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D layer with 32 filters, kernel size 3x3, ReLU activation
        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    model = MyModel()
    # Compile the model with the same parameters as reference example
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor representing a batch of grayscale images of size 28x28
    # The batch size is arbitrarily chosen as 64 for demonstration.
    # Input range and dtype match scale used in dataset loading (float32, 0-1 normalized).
    B = 64
    return tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)

