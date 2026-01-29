# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← MNIST grayscale images 28x28

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(300, activation='relu')
        self.dense2 = tf.keras.layers.Dense(100, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

def my_model_function():
    model = MyModel()
    # Compile model similar to the original code
    model.compile(
        loss="categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
    )
    return model

def GetInput():
    # MNIST input shape: batch size unknown, 28x28 grayscale images
    # Create a random batch of 32 images as example input
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

