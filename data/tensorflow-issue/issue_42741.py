# tf.random.uniform((B, 784), dtype=tf.float32)  <- Input shape inferred from SimpleModel's input_shape=(784,)

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replicating the SimpleModel architecture described in the issue
        # Dense(512, relu) -> Dropout(0.2) -> Dense(10)
        # input shape: (None, 784)
        self.layer_1 = keras.layers.Dense(512, activation='relu', input_shape=(784,))
        self.layer_2 = keras.layers.Dropout(0.2)
        self.layer_3 = keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.layer_1(inputs)
        x = self.layer_2(x, training=training)
        return self.layer_3(x)


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Typical compilation for classification task (not strictly required here but good practice)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def GetInput():
    # Create a random batch of input data with shape (B, 784)
    # Use batch size 32 as a common default
    batch_size = 32
    input_shape = (batch_size, 784)
    # Using float32 as it's standard for TF keras models
    return tf.random.uniform(input_shape, dtype=tf.float32)

