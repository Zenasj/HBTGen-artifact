# tf.random.uniform((None, 224, 224, 3), dtype=tf.float32) ← inferred input shape and dtype from NASNetMobile input_shape=(224,224,3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_labels=229, img_size=224):
        super().__init__()
        # Base NASNetMobile without top layers, frozen weights
        self.base_model = tf.keras.applications.NASNetMobile(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet')
        self.base_model.trainable = False
        
        # Pooling and dense output layer with sigmoid activation for multi-label classification
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(num_labels, activation='sigmoid', name="output")

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.global_pool(x)
        out = self.output_layer(x)
        return out

def my_model_function():
    """
    Returns an instance of MyModel initialized with default parameters.
    This replicates the model defined in the provided issue:
    NASNetMobile backbone (frozen) + GAP + Dense(229, sigmoid).
    """
    model = MyModel()
    # Optionally you could load pretrained weights here if available, e.g. model.load_weights(...)
    return model

def GetInput():
    """
    Returns a random tensor input conforming to the model's input expected shape:
    Batch size None (symbolic), height=224, width=224, channels=3, dtype float32.
    This matches the NASNetMobile input.
    """
    # Use a batch size 1 random input for testing
    import numpy as np
    input_shape = (1, 224, 224, 3)
    return tf.random.uniform(input_shape, dtype=tf.float32)

