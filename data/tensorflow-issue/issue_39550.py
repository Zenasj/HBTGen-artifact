# tf.random.uniform((1, 400, 400, 3), dtype=tf.float32) ← inferred input shape based on ImageDataGenerator target_size and batch size 1

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load InceptionV3 base model, exclude top layer, pretrained on ImageNet
        self.pretrained = InceptionV3(input_shape=(400, 400, 3), include_top=False, weights='imagenet')
        
        # Freeze layers before 'mixed1' (first few layers), others trainable
        trainable_flag = False
        for layer in self.pretrained.layers:
            if layer.name == 'mixed1':
                trainable_flag = True
            layer.trainable = trainable_flag
        
        # Extract output from 'mixed7' layer for custom classification head
        self.last_layer = self.pretrained.get_layer('mixed7').output
        
        # We'll redefine the model head inside __init__ using Functional API trick:
        # But since tf.keras.Model subclass requires layers defined, we'll build Dense layers here.
        
        # Since we subclass tf.keras.Model, define Dense layers here
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(4, activation='softmax')

        # We will build the head model once in __init__ for clarity
        # Define a Model from pretrained input to mixed7 output to share weights
        self.feature_extractor = Model(inputs=self.pretrained.input, outputs=self.pretrained.get_layer('mixed7').output)

    def call(self, inputs, training=False):
        x = self.feature_extractor(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel with pretrained InceptionV3 backbone
    return MyModel()

def GetInput():
    # Generate a random tensor matching input shape (batch_size=1, 400x400 RGB)
    # Values between 0 and 1 as the real data input is scaled (rescale=1/255 in generator)
    return tf.random.uniform((1, 400, 400, 3), dtype=tf.float32)

