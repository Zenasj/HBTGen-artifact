# tf.random.uniform((8, 10, 10, 3), dtype=tf.float32) ← input shape inferred from dataset created with imgs = tf.zeros([8, 10, 10, 3])

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, strategy=None):
        super().__init__()
        self.strategy = strategy
        # Conv layers as in Evaluation.__init__, created in strategy scope if provided
        if self.strategy:
            with self.strategy.scope():
                self.conv1 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same')
                self.conv2 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same')
        else:
            self.conv1 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same')
            self.conv2 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same')

    @tf.function
    def call(self, img):
        # Apply both conv layers to input
        pred1 = self.conv1(img)
        pred2 = self.conv2(img)
        return {'pre1': pred1, 'pre2': pred2}

def my_model_function(strategy=None):
    # Return an instance of MyModel with given strategy
    return MyModel(strategy=strategy)

def GetInput():
    # Return a random tensor matching the input shape [8,10,10,3], dtype float32
    # This matches the input dataset shape from the original code
    return tf.random.uniform((8, 10, 10, 3), dtype=tf.float32)

