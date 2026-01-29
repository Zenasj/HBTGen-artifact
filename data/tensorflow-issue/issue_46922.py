# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the original Sequential model used in the issue
        self.conv = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same', input_shape=(224, 224, 3))
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)
        
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel, mimicking the mock_model from the issue
    model = MyModel()
    # Build model to set input shapes correctly (necessary for TF < 2.4 sometimes)
    model.build(input_shape=(None, 224, 224, 3))
    return model


def GetInput():
    # Provides random float32 input tensor with the shape expected by MyModel (batch dim 1)
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

