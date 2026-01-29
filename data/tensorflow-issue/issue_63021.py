# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from the original example (100, 10) batch_size is dynamic

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Layer Normalization layer: keras.layers.Normalization with axis=-1 (default)
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        # Two Dense layers as in the original Sequential model
        self.dense1 = tf.keras.layers.Dense(10)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Forward pass mirrors original Sequential:
        # normalizer -> dense1 -> dense2
        x = self.normalizer(inputs, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate the model and adapt the Normalization layer with some dummy data.
    model = MyModel()
    
    # The Normalization layer requires adapt() to be called with representative data.
    # We'll use a small batch of random data matching the input shape for adaptation.
    dummy_adapt_data = tf.random.uniform((100, 10), dtype=tf.float32)
    model.normalizer.adapt(dummy_adapt_data)
    
    # Build the model by calling it once with the correct input shape
    _ = model(tf.zeros((1, 10)))

    return model

def GetInput():
    # Return a random input tensor matching shape (B, 10) where B=32 batch size as typical
    return tf.random.uniform((32, 10), dtype=tf.float32)

