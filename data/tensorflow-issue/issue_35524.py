# tf.random.uniform((None, 28, 28), dtype=tf.float32) ← Inferred input shape for the model (batch size is dynamic)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the model architecture identical to that in the issue:
        # Flatten input (28, 28) -> Dense 128 relu -> Dropout 0.2 -> Dense 10 softmax
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel; no pretrained weights as none provided.
    return MyModel()

def GetInput():
    # Return a random float32 tensor with shape (batch_size, 28, 28).
    # Use batch size 1 as default for simplicity.
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

