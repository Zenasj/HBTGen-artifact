# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from original model Input(shape=(28, 28, 1))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Conv2D layer with 2 filters, kernel_size=3, stride=2, ReLU activation
        self.conv1 = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=3,
            strides=2,
            activation='relu',
            padding='valid'  # padding not explicitly specified, default is 'valid'
        )
        
    def call(self, x):
        x = self.conv1(x)
        return x

def my_model_function():
    # Instantiate and build the model with input shape fixed
    model = MyModel()
    # Explicitly build the model to set weights shapes before use - batch size None for flexibility
    model.build(input_shape=(None, 28, 28, 1))
    return model

def GetInput():
    # Return a random tensor that matches the input expected by MyModel:
    # batch size = 1 (arbitrary), height=28, width=28, channels=1, dtype float32
    return tf.random.uniform(shape=(1, 28, 28, 1), dtype=tf.float32)

