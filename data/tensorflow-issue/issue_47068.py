# tf.random.uniform((1, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from the provided model and dataset

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture: Conv2D(128, 3, padding=same) + ReLU + Conv2D(3, 3, padding=same)
        self.conv1 = tf.keras.layers.Conv2D(128, 3, padding="same", name="Conv1")
        self.act1 = tf.keras.layers.ReLU(name="Act1")
        self.conv2 = tf.keras.layers.Conv2D(3, 3, padding="same", name="Conv2")

    def call(self, x):
        # Forward pass: expected input shape (B, 32, 32, 3) with dtype float32 normalized [0,1]
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x

def my_model_function():
    # Return an instance of the model
    model = MyModel()
    # Normally weights are loaded from a pretrained saved model; 
    # here we provide random initialization as placeholder.
    # To load weights from saved model "simpleconv", user would do:
    # saved = tf.keras.models.load_model("simpleconv")
    # model.set_weights(saved.get_weights())
    return model

def GetInput():
    # Create a random input tensor matching the model input: (1, 32, 32, 3), with float32 values normalized [0,1]
    # This matches the CIFAR-10 patch normalization used in the original example.
    return tf.random.uniform((1, 32, 32, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

