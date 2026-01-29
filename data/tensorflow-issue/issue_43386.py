# tf.random.uniform((1, IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float32) ← Input shape inferred from model's input_shape=(IMG_WIDTH, IMG_HEIGHT, 1) and grayscale channel=1

import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Assuming IMG_WIDTH and IMG_HEIGHT are predefined or placeholders
# Since the original issue does not specify exact dimensions,
# we will define them as constants here for completeness.
# These can be adjusted as needed.
IMG_WIDTH = 28
IMG_HEIGHT = 28

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Construct the Sequential model as described in the issue
        self.model = Sequential([
            layers.Conv2D(4, 3, padding='same', activation='relu',
                          input_shape=(IMG_WIDTH, IMG_HEIGHT, 1),
                          name='conv_layer1'),
            layers.MaxPooling2D(name='max_pooling1'),
            layers.Conv2D(4, 3, padding='same', activation='relu',
                          name='conv_layer2'),
            layers.MaxPooling2D(name='max_pooling2', pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(20, activation='relu', name='hidden_layer'),
            layers.Dense(1, activation='sigmoid', name='output')
        ])

    def call(self, inputs, training=False):
        return self.model(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape and dtype
    # Batch size 1, grayscale channel 1
    return tf.random.uniform((1, IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float32)

