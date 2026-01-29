# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple CNN model from the shared code, matches input shape (28,28,1)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits for 10 classes

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Returns a compiled instance of MyModel with sparse categorical crossentropy and Adam optimizer,
    # matching the setup in the original code snippet.
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Returns a batch of random inputs with shape (BATCH_SIZE, 28, 28, 1), dtype float32
    # BATCH_SIZE is inferred from the original issue: 64 per replica, total batch size depends on devices,
    # but we pick a typical batch size 64 here (single replica).
    batch_size = 64
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

