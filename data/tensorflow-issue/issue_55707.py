# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← input shape for single grayscale image batch

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Equivalent to the Sequential model in the issue:
        # Flatten 28x28 → 784 vector
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Dense hidden layer 128 units, relu activation
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Output logits layer - 10 classes
        self.dense2 = tf.keras.layers.Dense(10)
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model similarly to the issue's example
    # Use SparseCategoricalCrossentropy with from_logits=True,
    # Adam optimizer as in example (note: issue discusses optimizer bug on macOS metal, but here just setup model)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a batch of random inputs with shape (batch_size, 28, 28)
    # Use batch size 32 as a reasonable default
    batch_size = 32
    # Random float inputs normalized between 0 and 1 as in the preprocessing
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

