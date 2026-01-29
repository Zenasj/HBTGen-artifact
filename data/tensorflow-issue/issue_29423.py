# tf.random.uniform((32, 28, 28, 1), dtype=tf.float32) ← inferred input shape from MNIST dataset batching and model input shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        return self.dense(x)

def my_model_function():
    # Create and compile the model similarly to the reported code
    model = MyModel()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(0.01)
    )
    return model

def GetInput():
    # Return a random tensor mimicking a batch of MNIST images (batch_size=32)
    # Note: MNIST images are grayscale 28x28 with 1 channel
    # Use float32 as model input dtype
    return tf.random.uniform(shape=(32, 28, 28, 1), dtype=tf.float32)

