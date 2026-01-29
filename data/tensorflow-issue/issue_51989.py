# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST data preprocessing

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simplest model described in the issue: Flatten -> Dense(100) + ReLU -> Dense(120) + ReLU -> Dense(10)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(100)
        self.relu1 = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(120)
        self.relu2 = tf.keras.layers.ReLU()
        self.dense3 = tf.keras.layers.Dense(10)  # logits output for 10 classes

        # Loss and metrics defined in the original example
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        logits = self.dense3(x)
        return logits

    def compute_loss(self, x, y):
        logits = self.call(x, training=True)
        loss = self.loss_fn(y, logits)
        return loss, logits

    # This method is not requested but can be handy to replicate full behavior
    def get_metrics(self):
        return [self.accuracy]

def my_model_function():
    """
    Return an instance of MyModel, compiled with optimizer and loss to match the example.
    """
    model = MyModel()
    # Compile to ensure metrics etc. are correctly setup (matches example code)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model

def GetInput():
    """
    Return random input tensor shaped (batch, height, width, channels) like MNIST preprocessed data:
    batch size: 32 (arbitrary chosen for example)
    height & width: 28x28
    channels: 1 (grayscale)
    Input range preprocessed in example: float32 in range [-1, 1]
    """
    batch_size = 32
    x = tf.random.uniform(
        (batch_size, 28, 28, 1),
        minval=-1.0,
        maxval=1.0,
        dtype=tf.float32,
    )
    return x

