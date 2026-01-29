# tf.random.uniform((BATCH_SIZE, 28, 28, 3), dtype=tf.float32) ← Inferred input shape after preprocessing to 28x28 RGB images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Alexnet-like architecture adapted for CIFAR-10 input size 28x28x3 after preprocessing
        self.conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', name='conv1')
        self.pool1 = tf.keras.layers.MaxPool2D(2, 2, name='pool1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')

        self.conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', name='conv2')
        self.pool2 = tf.keras.layers.MaxPool2D(2, 2, name='pool2')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')

        self.conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', name='conv3')
        self.pool3 = tf.keras.layers.MaxPool2D(2, 2, name='pool3')
        self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')

        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.dropout = tf.keras.layers.Dropout(0.5)

        self.d1 = tf.keras.layers.Dense(1024, activation='relu', name='d1')
        self.d2 = tf.keras.layers.Dense(512, activation='relu', name='d2')
        self.d3 = tf.keras.layers.Dense(256, activation='relu', name='d3')
        self.out = tf.keras.layers.Dense(10, activation='softmax', name='out')

    def call(self, x, training=False):
        # Forward pass
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x, training=training)

        x = self.flatten(x)
        x = self.dropout(x, training=training)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.out(x)


def my_model_function():
    # Instantiate and return the Alexnet-based model
    return MyModel()


def GetInput():
    # Return a random input tensor matching the expected input:
    # shape: (BATCH_SIZE, 28, 28, 3)
    # dtype: float32
    # Using a reasonable batch size, e.g., 768 as in the original code.
    BATCH_SIZE = 768  # Example batch size (2 workers * 384 per worker)
    return tf.random.uniform(shape=(BATCH_SIZE, 28, 28, 3), dtype=tf.float32)

