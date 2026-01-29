# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← input shape inferred from MNIST dataset example

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import tensorflow.keras.backend as K

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Custom dense layer with internal variable 'a' to store intermediate tensor value as variable
        self.flatten = layers.Flatten()
        self.my_dense = MyDenseLayer(64)
        self.output_dense = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.my_dense(x)
        return self.output_dense(x)

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
        # Initialize variable to hold last output for inspection (matching issue suggestion)
        self.a = tf.Variable(tf.zeros([num_outputs]), trainable=False, dtype=tf.float32)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[int(input_shape[-1]), self.num_outputs],
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        # Compute the dot product 
        result = K.dot(inputs, self.kernel)
        # Assign to the variable to enable .numpy() access later
        self.a.assign(tf.reduce_mean(result, axis=0))  # reduce mean across batch to keep shape consistent
        return result

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape expected by MyModel i.e. (batch, 28, 28)
    # Assuming batch size of 32 for example
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

