# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=10, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Returns an instance of MyModel
    model = MyModel()
    # Compile with the same config as in the original example
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Generates a random batch of grayscale images with shape (batch_size, 28, 28, 1)
    # Using a batch size of 32 by default
    batch_size = 32
    return tf.random.uniform(shape=(batch_size, 28, 28, 1), dtype=tf.float32)

