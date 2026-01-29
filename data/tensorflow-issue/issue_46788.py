# tf.random.uniform((None, 28, 28), dtype=tf.float32) ← Input shape shape inferred from MNIST dataset (batch size None)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten input images of shape (28, 28) to vectors of length 784
        self.flatten = tf.keras.layers.Flatten()
        # Two hidden Dense layers with 128 units each and ReLU activation
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        # Output layer with 10 units for 10 classes with softmax activation
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of the Keras model as per the MNIST example
    model = MyModel()
    # Compile the model as in the posted example
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a batch of random MNIST-like grayscale images (batch size 32)
    # The model expects shape (batch_size, 28, 28), with pixel values normalized similar to the example
    batch_size = 32
    # Generate random floats (0-1) for grayscale images and then normalize along axis 1 to simulate input
    random_images = tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)
    # Normalize across axis 1 to mimic tf.keras.utils.normalize with axis=1 from the example
    # Add small epsilon to avoid division by zero
    norm = tf.norm(random_images, axis=1, keepdims=True) + 1e-7
    normalized_images = random_images / norm
    return normalized_images

