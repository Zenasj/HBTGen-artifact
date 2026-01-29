# tf.random.uniform((1024, 224, 224, 3), dtype=tf.float32) ← inferred input shape from example data (batch_size=1024)
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Input

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a small ConvNet replicating the given example model architecture
        # Input shape: (224, 224, 3)
        self.conv1 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', strides=2)
        self.conv2 = Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', strides=2)
        self.conv3 = Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', strides=1)
        self.conv4 = Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', strides=2)
        self.conv5 = Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', strides=2)
        self.conv6 = Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', strides=2)
        self.conv7 = Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', strides=2)
        self.gap = GlobalAveragePooling2D()
        self.fc = Dense(10)  # Output logits for 10 classes

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        out = self.fc(x)
        return out

def my_model_function():
    # Return an instance of MyModel. 
    # Note: Compilation (loss, optimizer) and TPU scope setup should be done outside this function,
    # as per the example issue code.
    return MyModel()

def GetInput():
    # Generate a dummy batch matching the example shape and dtype used in the issue:
    # 1024 samples of 224x224 RGB images (dtype float32).
    # Using random uniform data to resemble image inputs.
    batch_size = 1024
    height = 224
    width = 224
    channels = 3
    return tf.random.uniform(shape=(batch_size, height, width, channels), dtype=tf.float32)

