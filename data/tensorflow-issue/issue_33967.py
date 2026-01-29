# tf.random.uniform((32, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the MNIST CNN model described in the example
        # Input shape: (28, 28, 1)
        self.conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1))
        self.conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')
        self.pool = layers.MaxPooling2D(pool_size=(2,2))
        self.dropout1 = layers.Dropout(0.25)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu', name='features')  # Named 'features' for embeddings
        self.dropout2 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(10, activation='softmax')  # 10 classes for MNIST

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Instantiate the model and compile it with Adadelta optimizer and categorical_crossentropy
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a batch of random float32 tensors shaped like MNIST images
    # Batch size: 32 (matching typical batch) for concrete input to run in graph/JIT
    # Shape: (batch_size, height, width, channels)
    batch_size = 32
    img_rows, img_cols = 28, 28
    channels = 1
    return tf.random.uniform((batch_size, img_rows, img_cols, channels), dtype=tf.float32)

