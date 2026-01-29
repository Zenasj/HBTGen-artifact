# tf.random.uniform((1, 100), dtype=tf.float32) ← inferred input shape from model(tf.zeros([1, 100]))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model is a Sequential of two Dense layers with 100 units each, no activation specified
        self.layer1 = tf.keras.layers.Dense(100)
        self.layer2 = tf.keras.layers.Dense(100)
        
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x

def my_model_function():
    # Instantiate MyModel
    model = MyModel()
    # Build the model by calling once on dummy input to initialize weights
    _ = model(tf.zeros((1, 100)))
    return model

def GetInput():
    # Return a random tensor matching shape [1, 100] and dtype float32
    return tf.random.uniform(shape=(1, 100), dtype=tf.float32)

