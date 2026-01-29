# tf.random.uniform((B, num_features), dtype=tf.float32) ← Input is a 2D tensor with shape (batch_size, num_features)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

class MyModel(tf.keras.Model):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        # Build a simple MLP model similar to the example from the issue:
        # Input layer, Dense 64 ReLU, Dropout 0.2, Dense softmax output
        self.model = Sequential([
            Input(shape=(num_features,)),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

def my_model_function(num_features=10, num_classes=5):
    """
    Instantiate MyModel with default or specified input/output dimensions.
    Defaults chosen as reasonable guesses (10 features, 5 classes).
    """
    return MyModel(num_features=num_features, num_classes=num_classes)

def GetInput(batch_size=32, num_features=10):
    """
    Return a random input tensor consistent with the model input shape.
    - Batch size: variable (default 32)
    - Feature dimension: num_features (default 10)
    Data is random uniform float32 tensor.
    """
    return tf.random.uniform((batch_size, num_features), dtype=tf.float32)

