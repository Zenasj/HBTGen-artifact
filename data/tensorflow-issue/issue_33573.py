# tf.random.uniform((64, 32, 32, 3), dtype=tf.float32)  ← inferred from CIFAR-100 input shape (32x32 RGB images), batch size 64 used in example

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the architecture in the issue:
        # - Several Conv2D layers with BatchNorm + LeakyReLU
        # - Final conv layer expands channels to 512*100 as in original
        # - Flatten and Dense to intermediate 5*128 shape
        # - Reshape to (5,128) to feed an LSTM layer
        # - LSTM layer followed by flattening
        # - Final Dense to 100 classes with softmax activation
        
        self.conv1 = layers.Conv2D(128, 3, padding='same', strides=(2,2))
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.LeakyReLU(0.2)
        
        self.conv2 = layers.Conv2D(256, 3, padding='same', strides=(2,2))
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.LeakyReLU(0.2)
        
        self.conv3 = layers.Conv2D(512*100, 3, padding='same', strides=(2,2))
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.LeakyReLU(0.2)
        
        self.flatten1 = layers.Flatten()
        self.dense1 = layers.Dense(5*128)
        # Reshape to sequence for LSTM: (batch, 5, 128)
        self.reshape = layers.Reshape((5, 128))
        self.lstm = layers.LSTM(128)
        self.flatten2 = layers.Flatten()
        
        self.output_dense = layers.Dense(100, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.reshape(x)
        x = self.lstm(x)
        x = self.flatten2(x)
        
        return self.output_dense(x)

def my_model_function():
    # Instantiate and return the model as per the original architecture.
    model = MyModel()
    # The model requires an input shape to build weights; build the model
    # with input shape (64, 32, 32, 3) batch size 64 as used in example
    dummy_input = tf.random.uniform((64, 32, 32, 3), dtype=tf.float32)
    _ = model(dummy_input, training=False)
    return model

def GetInput():
    # Return a dummy tensor input that matches expected input shape (batch, height, width, channels)
    # CIFAR-100 images 32x32x3, batch size 64 is used in example data preparation
    return tf.random.uniform((64, 32, 32, 3), dtype=tf.float32)

