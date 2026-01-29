# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Inferred input shape from Fashion MNIST dataset (grayscale images 28x28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A small fashion MNIST style classifier similar to the ones described in the issue
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def my_model_function():
    # Construct an instance of MyModel
    model = MyModel()
    # Compile with adam optimizer and sparse categorical crossentropy as used in the examples
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Generate random input tensor with shape (batch_size=32, 28, 28), dtype float32
    # Values normalized between 0 and 1, matching preprocessing in the examples
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

