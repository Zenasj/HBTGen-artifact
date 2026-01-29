# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Based on model input shape (28, 28, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the convolutional layers similar to the Sequential example
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        return self.dense(x)

def my_model_function():
    # Returns an instance of MyModel.
    # Note: The original issue was the incorrect usage of optimizer as a tuple.
    # User should pass optimizer object directly, NOT in a singleton tuple.
    # Initialize and compile are not done here, since compile requires loss and optimizer
    # and are often done in training code, but a correct optimizer example:
    model = MyModel()
    optimizer = tf.keras.optimizers.SGD()  # Use optimizer object, not tuple
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy())
    return model

def GetInput():
    # Return random tensor of shape (batch_size=1, 28, 28, 1), dtype float32
    # Matches the input expected by MyModel.
    return tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)

