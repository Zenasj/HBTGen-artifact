# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Batch dimension B is dynamic, height=28, width=28, channels=1 grayscale image input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example CNN architecture from the GitHub issue
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Instantiate and compile model as per issue example
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Create a batch of random normalized grayscale images matching (B, 28, 28, 1)
    # Batch size 64 (arbitrary typical batch size used in example)
    batch_size = 64
    # Random uniform inputs resembling image pixels scaled to [0,1)
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

