# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape is (batch_size, 784) representing flattened MNIST images

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # First dense layer with relu activation
        self.first_dense = layers.Dense(64, activation='relu', name='dense_1')
        # Output dense layer with softmax activation for 10 classes
        self.out = layers.Dense(10, activation='softmax', name='predictions')

    def call(self, inp):
        # Forward pass: dense_1 -> predictions
        f_dense = self.first_dense(inp)
        s_dense = self.out(f_dense)
        return s_dense

    def input_receiver(self, inp):
        # Identity function, but placeholder for potential input processing
        return inp

    def response_receiver(self, output):
        # Identity function, placeholder for output processing
        return output

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32, name="serving")])
    def serve(self, request):
        # This function wraps the inference pipeline to provide a serving signature
        features = tf.identity(self.input_receiver(request), name='request')
        output = self.call(features)
        response = tf.identity(self.response_receiver(output), name='response')
        return response


def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()


def GetInput():
    # Create a random input tensor matching the expected input shape: (batch_size, 784) and dtype float32
    # We pick batch_size=1 as a minimal example
    batch_size = 1
    # Using uniform random values scaled to typical MNIST input range [0,1]
    return tf.random.uniform((batch_size, 784), dtype=tf.float32)

