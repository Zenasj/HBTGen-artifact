# tf.random.uniform((None, 28, 28, 1), dtype=tf.float32) ← inferred input shape for grayscale 28x28 MNIST images, batch size None (dynamic)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Layers roughly matching the functional_model from the issue,
        # including activity_regularizer and kernel_regularizer as L2 with l=0.0001
        self.conv1 = tf.keras.layers.Conv2D(
            32, 5, padding='same', activation='relu',
            activity_regularizer=tf.keras.regularizers.l2(l=0.0001),
            kernel_regularizer=tf.keras.regularizers.l2(l=0.0001),
            input_shape=(28, 28, 1))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        # BatchNorm commented out in original issue, so omitted here
        self.conv2 = tf.keras.layers.Conv2D(
            64, 5, padding='same', activation='relu',
            activity_regularizer=tf.keras.regularizers.l2(l=0.0001),
            kernel_regularizer=tf.keras.regularizers.l2(l=0.0001))
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Returns an instance of MyModel with required initialization.
    # Regularizers and shapes are included by default in the model definition.
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input shape of MyModel:
    # Based on TF default image data format 'channels_last', MNIST inputs are (28, 28, 1).
    # Batch dimension is dynamic, we choose batch=32 here.
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32, minval=0, maxval=1)

