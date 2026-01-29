# tf.random.uniform((None, 28, 28), dtype=tf.float32) ← Input shape inferred from the discussed Sequential model's input_shape=(28, 28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the model from the issue: Flatten -> Dense(512, relu) -> Dropout(0.2) -> Dense(10, softmax)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    # This model can be compiled and trained as in the original code snippet
    return MyModel()

def GetInput():
    # Return a random float32 tensor matching shape (batch_size, 28, 28)
    # Batch size arbitrary chosen as 32 for testing
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

