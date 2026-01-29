# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)  ← input shape inferred from CIFAR-10 dataset; B=batch size (dynamic)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the Sequential CNN from the issue with explicit layers
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # 10 classes for CIFAR-10

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Create and compile the model as exemplified in the issue
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor consistent with (batch_size, height, width, channels)
    # Batch size set to 32 as in the example batches
    batch_size = 32
    # Use dtype float32 to match typical image preprocessing (division by 255.0)
    input_tensor = tf.random.uniform((batch_size, 32, 32, 3), minval=0, maxval=1, dtype=tf.float32)
    return input_tensor

