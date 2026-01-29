# tf.random.uniform((32, 7, 9), dtype=tf.float32) ← inferred input shape from dataset batch and input shape [7,9]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using the model architecture that was in the example causing the shape inference issue:
        # Sequential with Dense(100, tanh) -> Flatten -> Dense(3, softmax)
        self.dense1 = tf.keras.layers.Dense(100, activation='tanh', input_shape=(7, 9))
        self.flatten = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.flatten(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate and compile the model with Adam optimizer and categorical cross-entropy loss,
    # matching the original example setup.
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0),
        loss=tf.keras.losses.CategoricalCrossentropy()
    )
    return model

def GetInput():
    # Return a random tensor input matching the expected input shape:
    # batch size 32, shape (7, 9), dtype float32.
    return tf.random.uniform(shape=(32, 7, 9), dtype=tf.float32)

