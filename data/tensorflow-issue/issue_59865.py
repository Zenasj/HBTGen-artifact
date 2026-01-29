# tf.random.uniform((32, 224, 224, 3), dtype=tf.float32) ← input shape inferred from original example

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize two copies of VGG16 without top and pretrained weights,
        # set trainable=False to match original code behavior
        # Naming them explicitly to avoid Keras layer name clashes
        with tf.device("/gpu:0"):
            self.vgg1 = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
            self.vgg1._name = "vgg1"  # unique name
            self.vgg1.trainable = False
        with tf.device("/gpu:1"):
            self.vgg2 = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
            self.vgg2._name = "vgg2"  # unique name
            self.vgg2.trainable = False
        
        # Global average pooling layer (used on CPU per original model)
        # Just instantiate once, call on CPU device in call()
        self.gap = GlobalAveragePooling2D()
        # Final dense layer for classification
        self.final_dense = Dense(10)
    
    @tf.function(jit_compile=True)
    def call(self, images):
        # Manually split batch into two halves (data parallel inputs)
        # The original uses split on axis 0 (batch)
        first, second = tf.split(images, num_or_size_splits=2, axis=0)
        
        # Run the first half on GPU:0 with first VGG16 model
        with tf.device("/gpu:0"):
            vgged1 = self.vgg1(first)
        
        # Run the second half on GPU:1 with second VGG16 model
        with tf.device("/gpu:1"):
            vgged2 = self.vgg2(second)
        
        # Move data back to CPU for pooling and concatenation
        with tf.device("/cpu:0"):
            gap1 = self.gap(vgged1)
            gap2 = self.gap(vgged2)
            concatenated = tf.concat([gap1, gap2], axis=1)
            out = self.final_dense(concatenated)
        return out

def my_model_function():
    # Create and return a new instance of MyModel
    return MyModel()

def GetInput():
    # Generate a batch of 32 random images of shape (224, 224, 3)
    # dtype float32 to match typical VGG16 inputs
    return tf.random.uniform((32, 224, 224, 3), dtype=tf.float32)

