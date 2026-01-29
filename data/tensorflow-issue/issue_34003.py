# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST dataset (28x28 grayscale images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Flatten layer to convert 2D image input to 1D vector
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Dense layer with 128 units and ReLU activation
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Dropout layer with 0.2 dropout rate (during training)
        self.dropout = tf.keras.layers.Dropout(0.2)
        # Output layer with 10 units (digits 0-9) and softmax activation
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor with shape (batch_size=32, height=28, width=28)
    # scaled to [0,1] to simulate normalized MNIST images.
    # Batch size 32 is a common training batch size.
    return tf.random.uniform((32, 28, 28), minval=0, maxval=1, dtype=tf.float32)

