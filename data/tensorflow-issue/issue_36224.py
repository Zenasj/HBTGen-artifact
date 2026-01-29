# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        # Define a simple CNN model similar to the MNIST example in the issue
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return the model instance compiled ready for training
    model = MyModel()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return random inputs with shape [batch, height, width, channels]
    # Batch size chosen to reflect batched usage; assuming 128 as example (64 per replica * 2)
    batch_size = 128
    # Random float32 tensor normalized to [0,1] matching scaled MNIST input
    x = tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)
    return x

