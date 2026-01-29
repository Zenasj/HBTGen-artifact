# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape for MNIST grayscale images 28x28

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))  # Flatten 28x28 images
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.02, seed=1)
        self.dense2 = tf.keras.layers.Dense(254, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.02, seed=1)
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.01, seed=1)
        self.dense4 = tf.keras.layers.Dense(64, activation='relu')
        self.dense5 = tf.keras.layers.Dense(32, activation='sigmoid')
        self.output_layer = tf.keras.layers.Dense(10)  # 10 classes for MNIST

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.output_layer(x)


def my_model_function():
    # Instantiate model and compile as per example
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Generate a random batch of grayscale MNIST-like images (batch size 32)
    # Values between 0 and 1, float32.
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

