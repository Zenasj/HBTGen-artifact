# tf.random.uniform((BATCH_SIZE, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers

# Based on the issue, the user tries to augment batches of images shaped (batch_size, height, width, channels)
# The main problem is caused by applying dataset.batch() before augmentation, making augmentation receive 5-D tensors
# (batch dimension + another unexpected dimension). To fix this, augmentation layers must only receive 4-D tensors:
# (batch_size, height, width, channels). So here, we implement a model that expects input shape (None, 224, 224, 3).

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Rescaling layer to normalize pixels to [0,1]
        self.rescale = layers.Rescaling(1./255)
        # Data augmentation sequential layers, configured for 4D tensor inputs (NHWC)
        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])

        # For demonstration, add a simple conv layer (not stated in issue, but to make model do something)
        self.conv = layers.Conv2D(16, 3, padding='same', activation='relu')
        self.pool = layers.MaxPool2D()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(10)  # Assume 10 classes or output classes for example

    def call(self, inputs, training=False):
        # Expect inputs with shape (batch_size, 224, 224, 3)
        x = self.rescale(inputs)
        if training:
            # Apply augmentation only in training mode
            x = self.data_augmentation(x, training=training)
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel with all layers initialized
    return MyModel()

def GetInput():
    # Generate a random tensor of shape (batch_size=18, height=224, width=224, channels=3)
    # This matches the input expected after batching for the model as per the issue
    batch_size = 18  # use same as BATCH_SIZE used in the issue examples
    height, width, channels = 224, 224, 3
    # Use uniform random floats between 0 and 255 to mimic raw image input pixels
    # dtype=tf.float32 because keras layers expect float inputs
    return tf.random.uniform(
        (batch_size, height, width, channels),
        minval=0,
        maxval=255,
        dtype=tf.float32
    )

