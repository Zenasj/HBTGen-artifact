# tf.random.uniform((B, 224), dtype=tf.float32) ← Inferred input shape based on example input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create input layer with explicit name 'Image_input'
        self.input_layer = tf.keras.layers.Input(shape=(224,), name='Image_input', dtype=tf.float32)
        # Dense layer outputting 5 units
        self.dense = tf.keras.layers.Dense(5)
        # Softmax activation layer with explicit name 'output-softmax'
        self.softmax = tf.keras.layers.Activation('softmax', name='output-softmax')

    def call(self, inputs):
        # Forward pass through the model implementing input->dense->softmax
        x = self.dense(inputs)
        return self.softmax(x)

def my_model_function():
    # Return instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching input shape (batch size flexible, 224)
    # Use batch size 16 as example for typical batch processing
    # dtype matches input layer dtype: float32
    return tf.random.uniform((16, 224), dtype=tf.float32)

