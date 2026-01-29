# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← MNIST images shape used in example code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=[28, 28])
        self.dense = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.flatten(inputs)
        return self.dense(x)

def my_model_function():
    """
    Returns an instance of MyModel, compiled with sparse categorical crossentropy loss,
    SGD optimizer and accuracy metric. This matches the minimal MNIST example code in the issue.
    """
    model = MyModel()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.SGD(),
        metrics=["accuracy"]
    )
    return model

def GetInput():
    """
    Returns a random tensor simulating a batch of MNIST grayscale images (shape: B x 28 x 28).
    Batch size is chosen as 32 as a reasonable default for testing.
    """
    batch_size = 32
    # Random floats between 0 and 1 simulating normalized MNIST images
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

