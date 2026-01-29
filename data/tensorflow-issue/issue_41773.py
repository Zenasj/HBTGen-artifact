# tf.random.uniform((32, 28, 28), dtype=tf.float32) ← inferred input shape from MNIST batch size 32 and image shape (28, 28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple feedforward model as per the described example
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Instantiate and compile the model with the specified optimizer and loss consistent with the example
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Generate a random input tensor that matches the input shape expected by the model: batch size 32, 28x28 images
    # Use float32 as dtype, scaled similarly to MNIST normalized images
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

