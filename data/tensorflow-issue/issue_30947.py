# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32) ← Input shape and dtype inferred from MNIST dataset preprocessing in the original example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model layers mimic the original Keras Sequential in the issue
        # Note: The original model used a BatchNormalization layer which caused bug with Estimator + MirroredStrategy.
        # We'll keep it here to match the original model structure, assuming no workaround needed here.
        # If needed, comment out BN layer to avoid bug as discussed in the issue comments.
        self.conv = tf.keras.layers.Conv2D(
            32, 3, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.02),
            input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.batchnorm(x, training=training)
        return self.dense2(x)


def my_model_function():
    # Returns a compiled instance of MyModel with Adam optimizer and appropriate loss/metrics
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Returns a random batch of inputs consistent with MNIST shape (BATCH_SIZE, 28, 28, 1)
    # Use batch size 64 as in original example
    BATCH_SIZE = 64
    # Input values scaled in [0,1] float32 as in scaling function in original input_fn
    return tf.random.uniform(shape=(BATCH_SIZE, 28, 28, 1), dtype=tf.float32, minval=0.0, maxval=1.0)

