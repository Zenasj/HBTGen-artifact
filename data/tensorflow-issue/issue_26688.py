# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← We infer input shape as (batch_size, 28, 28) grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Model defined from the example sequential model provided in the issue
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        # Use dropout conditionally on training mode
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor of shape (batch_size, 28, 28)
    # Assuming batch size of 32 for the input tensor
    batch_size = 32
    # Use uniform random values between 0 and 1 (typical for normalized pixel images)
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

