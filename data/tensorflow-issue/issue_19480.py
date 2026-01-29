# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← assumed input shape for image data (e.g., (None, 32, 32, 3))

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A Keras model illustrating a minimal Conv2D -> Flatten -> Dense topology,
    encapsulated for testing device placement issues related to resource variable access.
    
    This reflects the minimal reproducible example from the issue where placing the model
    explicitly on a CPU device but trying to run with GPU execution causes FailedPreconditionError.
    
    The model input shape is assumed (batch_size, 32, 32, 3) matching CIFAR-10 size images.
    """

    def __init__(self):
        super().__init__()
        # Conv2D layer: 32 filters, 3x3 kernel
        self.conv = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))
        # Flatten for dense layer input
        self.flatten = tf.keras.layers.Flatten()
        # Output dense layer with 10 units for classification
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        return self.dense(x)

def my_model_function():
    """
    Creates and returns an instance of MyModel.
    
    Notes:
    - No pre-trained weights are loaded.
    - The model is compiled and ready for training/testing.
    """
    model = MyModel()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def GetInput():
    """
    Returns a random input tensor matching the expected input shape of MyModel.
    
    Assumptions:
    - Batch size is chosen as 16 as a reasonable default for testing.
    - Input shape is (16, 32, 32, 3) of dtype float32, as CIFAR-10 style images are float32.
    """
    # Shape: (batch_size=16, height=32, width=32, channels=3)
    return tf.random.uniform((16, 32, 32, 3), dtype=tf.float32)

