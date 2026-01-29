# tf.random.uniform((B, 1, 256, 256, 64), dtype=tf.float32) ← inferred input shape from data generator and printed shapes

import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # The input shape inferred from data generator and prints is (B, 1, 256, 256, 64)
        # Define 3D convolutional layers and pooling layers as per original MultiBranchCNN
        self.conv3d = Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same')
        self.maxpool3d = MaxPool3D(pool_size=(3,3,3))
        self.conv3d2 = Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same')
        self.maxpool3d2 = MaxPool3D(pool_size=(3,3,3))
        self.conv3d3 = Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same')
        self.maxpool3d3 = MaxPool3D(pool_size=(3,3,3))
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dropout = Dropout(0.1)
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Cast input to float32 to avoid type issues
        x = tf.cast(inputs, tf.float32)
        x = self.conv3d(x)
        x = self.maxpool3d(x)
        x = self.conv3d2(x)
        x = self.maxpool3d2(x)
        x = self.conv3d3(x)
        x = self.maxpool3d3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor that matches the batch dimension and model input shape
    # Since batch size in the example code was often 4 in debug logs, we use 4 here
    batch_size = 4
    # Shape: (batch_size, 1, 256, 256, 64)
    # Use float32 values as typical for images after normalization
    input_tensor = tf.random.uniform((batch_size, 1, 256, 256, 64), dtype=tf.float32)
    return input_tensor

