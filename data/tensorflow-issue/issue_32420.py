# tf.random.normal((1, 100), dtype=tf.float32)  ← Input shape inferred from the issue (x shape = (1,100))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A single Dense layer with 100 units and softmax activation, as in the example
        self.dense = tf.keras.layers.Dense(100, activation=tf.nn.softmax)
    
    def call(self, inputs):
        # Forward pass: apply dense layer
        return self.dense(inputs)

def my_model_function():
    # Instantiate MyModel and compile it to replicate the example behavior
    model = MyModel()
    model.compile(loss='mse', optimizer='sgd')
    return model

def GetInput():
    # Return an input tensor shaped (1,100), matching the example input
    # Using tf.random.normal to generate random normal data as per the issue code
    return tf.random.normal((1, 100), dtype=tf.float32)

