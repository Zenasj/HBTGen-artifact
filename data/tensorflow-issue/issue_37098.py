# tf.random.uniform((B, 180, 320, 3), dtype=tf.float32) ← Input shape inferred from IMG_HEIGHT=180, IMG_WIDTH=320, 3 channels (RGB)

import tensorflow as tf
from tensorflow.keras import layers, models, losses

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a CNN model as described in the issue:
        # Conv2D(4 filters, 3x3) -> MaxPooling(2x2) -> Conv2D(8 filters, 3x3) -> MaxPooling(2x2)
        # Flatten -> Dense(3 units, sigmoid)
        self.conv1 = layers.Conv2D(4, (3, 3), activation='relu', input_shape=(180, 320, 3))
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(8, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(3, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        return self.dense(x)

def my_model_function():
    # Instantiate the model and compile with the same optimizer, loss, and metrics as in the reported code
    model = MyModel()
    # Compile model similarly; note that "categorical_crossentropy" was used in issue,
    # but labels seem multi-label ([0,0,0,1] style?). Assuming multi-class classification with 3 output units,
    # using categorical crossentropy as in original code.
    model.compile(
        optimizer='adam',
        loss=losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random float tensor simulating a batch of images with proper shape
    # Batch size assumed from example (e.g. 32). This matches original batch_size.
    BATCH_SIZE = 32
    IMG_HEIGHT = 180
    IMG_WIDTH = 320
    CHANNELS = 3
    return tf.random.uniform((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=tf.float32)

