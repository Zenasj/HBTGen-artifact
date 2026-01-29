# tf.random.uniform((64, 28, 28, 1), dtype=tf.float32) ← inferred input shape and dtype from the MNIST dataset preprocessing in input_fn

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Same architecture as the `make_model()` function from the issue:
        self.conv2d = tf.keras.layers.Conv2D(
            32, 3, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.02),
            input_shape=(28, 28, 1)
        )
        self.maxpool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv2d(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.batchnorm(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel.
    # Weights are randomly initialized by default, similar to original.
    model = MyModel()
    # Optionally compile the model to align with original usage pattern.
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Produce a batch of random images in expected shape (batch, height, width, channels)
    # dtype float32 between 0 and 1 as scaled in original input_fn
    return tf.random.uniform(shape=(64, 28, 28, 1), dtype=tf.float32)

