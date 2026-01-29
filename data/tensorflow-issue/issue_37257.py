# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred input shape for MNIST-like data

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the MNIST simple MLP model from the issue comments
        # Flatten input 28x28 to 784
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Dense layer 128 units relu
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Dropout 0.2
        self.dropout = tf.keras.layers.Dropout(0.2)
        # Final Dense layer 10 units (class logits)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate the model; weights are randomly initialized (typical in TF Keras)
    # No pretrained weights provided in the issue
    return MyModel()

def GetInput():
    # Return a random input tensor matching the input expected by MyModel
    # Batch size 1 (like used in the issue example), 28x28 image float32 scaled between 0 and 1
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

