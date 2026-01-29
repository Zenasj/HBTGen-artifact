# tf.random.uniform((global_batch_size, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST dataset and batch dimensions

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Build a CNN model similar to the one described in the issue examples
        self.reshape = tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        self.conv1 = tf.keras.layers.Conv2D(256, 2, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(128, 2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, 1, activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(32, 2, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(2048, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(10)  # No activation, for logits output

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        logits = self.logits(x)
        return logits

def my_model_function():
    # Return an instance of MyModel, compiled similarly to the example workflows
    model = MyModel()
    # Compile with SparseCategoricalCrossentropy from_logits since output layer logits are unscaled
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

def GetInput():
    # Return a random float32 tensor shaped as a batch of grayscale MNIST images
    # The batch size here is chosen as 64 by default; can be scaled by number of workers outside
    batch_size = 64  
    # MNIST images are 28x28 grayscale (single channel)
    return tf.random.uniform(shape=(batch_size, 28, 28), dtype=tf.float32)

