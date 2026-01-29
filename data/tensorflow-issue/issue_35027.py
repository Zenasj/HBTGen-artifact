# tf.random.uniform((128, 28, 28, 1), dtype=tf.float32) ← inferred from MNIST dataset batch size and image shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Parameters based on the original MNIST ConvNet example
        filters = 56
        kernel_size = 5
        units = 24

        # Build layers roughly matching the original sequential model
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=(kernel_size, kernel_size),
                                           activation='relu',
                                           input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Instantiate and compile the model using the same settings as the issue snippet
    model = MyModel()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random input tensor simulating a batch of MNIST grayscale images
    # Shape: (BATCH_SIZE=128, Height=28, Width=28, Channels=1)
    # dtype float32 as expected by the model
    batch_size = 128
    height = 28
    width = 28
    channels = 1
    # Uniform values in [0,1) simulating scaled pixel intensities as done in the original preprocessing
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

