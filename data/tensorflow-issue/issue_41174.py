# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape is (batch_size, 784) for flattened MNIST images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer as per the example with 1 output unit
        self.dense = tf.keras.layers.Dense(1, input_shape=(784,))

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Instantiate and compile the model as in the example
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model

def GetInput():
    # Return random input tensor matching (batch_size, 784), here batch_size=32 arbitrarily chosen
    return tf.random.uniform((32, 784), dtype=tf.float32)

