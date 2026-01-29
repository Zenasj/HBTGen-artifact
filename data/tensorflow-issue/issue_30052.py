# tf.random.uniform((64, 784), dtype=tf.float32) ← inferred from batching of (x_train, y_train) with batch size 64 and input shape (784,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Equivalent to the provided functional API model:
        # Input shape (784,)
        # Dense(64, relu) -> Dense(64, relu) -> Dense(10, softmax)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


def my_model_function():
    # Create and compile model as in the issue
    model = MyModel()
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return a random tensor batch simulating mnist flattened input
    # batch size 64, input size 784
    return tf.random.uniform((64, 784), dtype=tf.float32)

