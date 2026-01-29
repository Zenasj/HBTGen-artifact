# tf.random.uniform((BATCH_SIZE, 16, 16), dtype=tf.float32)
import tensorflow as tf
import numpy as np

BATCH_SIZE = 128
FEATURE_SHAPE = (16, 16)
PREFETCH_FACTOR = 4


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A single trainable variable simulating weights for linear model on feature tensor of shape (16,16)
        self.weights = tf.Variable(tf.random.normal(FEATURE_SHAPE), name="weights")
        # An optimizer instance
        self.optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def call(self, batch_x):
        # Linear prediction: sum over elementwise multiplication of weights and batch_x
        return tf.reduce_sum(self.weights * batch_x, axis=[1, 2])

    @tf.function
    def train_step(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            predictions = self.call(batch_x)
            loss = tf.reduce_mean(tf.square(batch_y - predictions))
        grads = tape.gradient(loss, [self.weights])
        self.optimizer.apply_gradients(zip(grads, [self.weights]))
        return loss

    def train(self, batch_x, batch_y):
        # Run training step and return loss as numpy float32
        loss = self.train_step(batch_x, batch_y)
        return loss


def my_model_function():
    # Return a new instance of the model (weights initialized randomly)
    return MyModel()


def GetInput():
    # Returns a random tensor of shape (BATCH_SIZE, 16, 16) matching the input expected by the model
    # dtype is tf.float32 as the weights and computation expect float32
    return tf.random.uniform((BATCH_SIZE, 16, 16), dtype=tf.float32)

