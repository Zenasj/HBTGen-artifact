# tf.random.uniform((100, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from the provided code snippet

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the same Sequential model layers as given in the issue
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # num_classes=10

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile model with the optimizer, loss and metrics as per the issue's example
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor simulating the input shape used in the issue
    # The input tensor is (100, 28, 28, 1) float32 as per the original training data
    return tf.random.uniform((100, 28, 28, 1), dtype=tf.float32)

