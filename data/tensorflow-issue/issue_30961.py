# tf.random.uniform((100, 28, 28), dtype=tf.float32) ← batch size 100, 28x28 grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture matching the reported example:
        # Flatten input [28,28], Dense 300 relu, Dense 100 relu, Dense 10 softmax
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(300, activation="relu")
        self.dense2 = tf.keras.layers.Dense(100, activation="relu")
        self.output_layer = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=False):
        # Forward pass
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

def my_model_function():
    # Return a fresh MyModel instance compiled with standard sparse categorical crossentropy and Adam optimizer,
    # matching the training setup described in the issue.
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random batch of grayscale images simulating Fashion MNIST input.
    # Use float32 type and scale to [0,1] to follow recommendation from the issue discussion.
    return tf.random.uniform((100, 28, 28), minval=0, maxval=1, dtype=tf.float32)

