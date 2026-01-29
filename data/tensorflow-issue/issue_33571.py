# tf.random.uniform((64, 28, 28, 1), dtype=tf.float32) ← Inferred input shape from model input_shape of (28,28,1) and batch size 64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the ConvNet layers as given in the example
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
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model similar to the original example setup
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a batch of images matching the expected input shape (batch_size=64, 28x28 grayscale)
    # Simulate normalized image input as float32 in [0,1]
    return tf.random.uniform((64, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)

