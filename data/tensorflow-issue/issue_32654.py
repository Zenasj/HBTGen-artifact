# tf.random.uniform((12, 372, 558, 3), dtype=tf.float32) ← inferred input shape and dtype from issue batch size and features_shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using tf.keras.applications.DenseNet121 configured as per the original issue
        # with no weights, input_shape (372, 558, 3), and 10 output classes.
        self.model = tf.keras.applications.DenseNet121(
            weights=None,
            input_shape=(372, 558, 3),
            classes=10,
        )
        # Explicitly build the model with batch size 12 and input shape
        self.model.build((12, 372, 558, 3))
        
    def call(self, inputs, training=False):
        # Forward pass through DenseNet121
        return self.model(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the expected input shape of MyModel
    # Batch size 12, Height 372, Width 558, Channels 3
    return tf.random.uniform((12, 372, 558, 3), dtype=tf.float32)

