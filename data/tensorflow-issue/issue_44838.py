# tf.random.uniform((B, 300, 480, 3), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.applications import Xception

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Input shape assumed from original example: (300, 480, 3)
        inp = tf.keras.Input(shape=(300, 480, 3))
        
        # Load pretrained Xception base model without top
        self.base_model = Xception(include_top=False, weights='imagenet',
                                  input_tensor=inp, pooling='avg')
        
        # Freeze layers up to layer 65, fine-tune afterwards - per original code
        for layer in self.base_model.layers[:65]:
            layer.trainable = False
        for layer in self.base_model.layers[65:]:
            layer.trainable = True
        
        # Define additional layers from original user model structure
        self.base_dropout = Dropout(0.5, name='base_drop')
        self.dense_alpha = Dense(1024, activation='relu', name='dense_alpha')
        self.dense_alpha_dropout = Dropout(0.5, name='dense_alpha_drop')
        self.dense_out = Dense(2, activation='softmax', name='dense_out')
        
        # Store the input
        self._input = inp

    def call(self, inputs, training=False):
        # Forward pass through base model to get features
        x = self.base_model(inputs, training=training)
        # Base dropout after feature extractor
        x = self.base_dropout(x, training=training)
        # Pass through dense layers
        x = self.dense_alpha(x)
        x = self.dense_alpha_dropout(x, training=training)
        out = self.dense_out(x)
        return out
    
    # Provide method to get base model output (base + dropout)
    def base_output(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.base_dropout(x, training=training)
        return x

    # Provide method to get second half output (dense_alpha input -> dense_out output)
    def second_half(self, features, training=False):
        x = self.dense_alpha(features)
        x = self.dense_alpha_dropout(x, training=training)
        out = self.dense_out(x)
        return out

def my_model_function():
    # Instantiate the full model
    return MyModel()

def GetInput():
    # Generate a random input tensor matching (B, 300, 480, 3)
    # Assume batch size = 1 for testing
    return tf.random.uniform((1, 300, 480, 3), dtype=tf.float32)

