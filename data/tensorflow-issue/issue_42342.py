# tf.random.uniform((1, 200, 200, 3), dtype=tf.float32) ← The model is defined with input_shape=(200, 200, 3).
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras import Model

# Assumption:
# - The input images are 200x200 RGB images (3 channels).
# - num_class is not specified; guessing a typical number like 10 for demonstration.
# - The issue was caused by feeding 256x256 images to a model expecting 200x200 inputs.
# - This code reconstructs the original model with corrected input size.
# - The output is a softmax classification over num_class categories.

num_class = 10  # inferred; replace with actual number of classes if known

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load ResNet50 without top layers, input shape (200,200,3)
        self.base_model = ResNet50(
            weights=None,
            include_top=False,
            input_shape=(200, 200, 3)
        )
        self.flatten = Flatten()
        self.dropout = Dropout(0.2)
        self.dense1 = Dense(32, activation='relu')
        self.dense2 = Dense(16, activation='relu')
        self.predictions = Dense(num_class, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.predictions(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching input shape (1, 200, 200, 3)
    # Batch size 1 for predict
    return tf.random.uniform((1, 200, 200, 3), dtype=tf.float32)

