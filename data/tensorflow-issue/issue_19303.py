# tf.random.uniform((B, 1), dtype=tf.string) ← Input is a batch of single string elements with shape (batch_size, 1)

import tensorflow as tf
from tensorflow.keras import layers, Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Using the Universal Sentence Encoder from TF Hub as in the original code,
        # but here we simulate the embedding layer because TF Hub usage requires session, etc.
        # We'll create a placeholder embedding layer using a Dense layer for demonstration.
        # In practice, replace with a TF Hub USE layer or appropriate embedding.
        self.embedding_dim = 512
        
        # Placeholder embedding simulating USE embedding output
        self.embedding_layer = layers.Lambda(
            lambda x: tf.zeros((tf.shape(x)[0], self.embedding_dim)), 
            output_shape=(self.embedding_dim,))
        
        self.dense1 = layers.Dense(1024, activation='relu')
        self.batchnorm = layers.BatchNormalization()
        self.prediction = layers.Dense(2000, activation='softmax')
        
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, 1), dtype=tf.string
        # In the original code, they do:
        # tf.squeeze(tf.cast(x, tf.string)) → remove last dimension and enforce string dtype
        # Here inputs should be shape (batch_size,1) dtype string
        # For demonstration, just pass inputs to embedding (which is placeholder zeros)
        
        # The real USEEmbedding function needs TF Hub and session calls which are not compatible here
        # So we simulate embedding output with zeros directly:
        x = self.embedding_layer(inputs)  # (batch_size, 512)
        x = self.dense1(x)
        x = self.batchnorm(x, training=training)
        x = self.prediction(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a batch of random tf.string tensors shaped (batch_size, 1)
    # We cannot generate random strings easily, so use fixed strings for illustration
    batch_size = 4  # small batch for demonstration
    # Tensor of shape (batch_size, 1), dtype string
    example_strings = tf.constant([["hello world"], ["tensorflow keras"], ["test input"], ["sample text"]])
    return example_strings

