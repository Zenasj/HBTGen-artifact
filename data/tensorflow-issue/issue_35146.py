# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape corresponds to flattened 28x28 MNIST images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the model layers as per the issue reproduction code
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu', name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(64, activation='relu', name='dense_2')
        self.predictions = tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inputs, training=False):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.predictions(x)

def my_model_function():
    # Instantiate and compile the model exactly as in the provided code
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

def GetInput():
    # Generate a random float32 tensor matching input: batch size unspecified, flattened 784 features
    # Use batch size 32 as a reasonable default for example usage
    return tf.random.uniform(shape=(32, 784), dtype=tf.float32)

