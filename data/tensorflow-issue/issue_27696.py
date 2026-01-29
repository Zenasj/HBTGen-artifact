# tf.random.uniform((64, 28, 28, 1), dtype=tf.float32) ← Inferred input shape & dtype based on MNIST example with batch size 64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example model architecture described in issue's code
        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                                           input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Instantiate the model, compile with a DistributionStrategy-compatible optimizer (tf.keras Adam)
    model = MyModel()

    # Important: to avoid the "Only TensorFlow native optimizers are supported" error with MirroredStrategy,
    # use an actual tf.keras optimizer (which is native), and instantiate compilation inside the training scope.
    # Adam with default parameter compatible with TF native optimizer support.
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Create a batch of 64 images of shape 28x28x1, dtype float32, values between 0 and 1 (scaled pixel range)
    return tf.random.uniform((64, 28, 28, 1), dtype=tf.float32)

# Notes and assumptions based on the issue:
# - Input shape (28, 28, 1) is inferred from MNIST example
# - Batch size 64 is used as in the example input_fn batch size
# - Loss is sparse_categorical_crossentropy, matching label shape [batch, 1] (integer class indices)
# - Model compiles with tf.keras.optimizers.Adam (the native TF optimizer class)
# - This mirrors what was required to avoid the "Only TensorFlow native optimizers are supported" error in
#   DistributedStrategy context
# - The model architecture is taken from the Keras Sequential example in the issue, transformed to subclassing style
# - Dropout and BatchNormalization layers respect the 'training' flag in call()

