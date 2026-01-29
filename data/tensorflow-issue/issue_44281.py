# tf.random.uniform((16, 224, 224, 3), dtype=tf.float32) ← inferred input shape used in original example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recap from the issue: The user wants a model like Xception without pretrained weights,
        # with input shape (224, 224, 3) and 1000 classes.
        # We'll embed Xception as a submodel inside MyModel.
        self.base_model = tf.keras.applications.Xception(
            weights=None,
            input_shape=(224, 224, 3),
            classes=1000
        )

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

def my_model_function():
    # Initialize the MyModel instance with Xception base model.
    return MyModel()

def GetInput():
    # Return a random tensor input matching input shape expected by MyModel.
    # Batch size 16 chosen per original snippet batch_size=16.
    return tf.random.uniform(shape=(16, 224, 224, 3), dtype=tf.float32)

