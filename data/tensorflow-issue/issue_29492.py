# tf.random.uniform((16, 224, 224, 3), dtype=tf.float32)  ← Assumed input shape based on common image input for DenseNet-like models

import tensorflow as tf
from tensorflow.keras import layers

class BottleneckLayer(layers.Layer):
    def __init__(self, growthRate):
        super().__init__()
        # 1x1 conv expands the channels to 4 * growthRate
        self.conv1 = layers.Conv2D(4 * growthRate, kernel_size=1, strides=1, padding="same")
        # 3x3 conv reduces channels to growthRate
        self.conv2 = layers.Conv2D(growthRate, kernel_size=3, strides=1, padding="same")
        self.batchNorm1 = layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.batchNorm2 = layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.relu = layers.Activation("relu")

    def call(self, x):
        # BN-ReLU-Conv1
        y = self.batchNorm1(self.relu(self.conv1(x)))
        # BN-ReLU-Conv2
        y = self.batchNorm2(self.relu(self.conv2(y)))
        # Concatenate input and output channels
        y = tf.concat([x, y], axis=-1)
        return y

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        growthRate = 12
        self.relu = layers.Activation("relu")
        # Initial conv layer (7x7 conv with stride 2, padding same)
        self.conv1 = layers.Conv2D(2 * growthRate, kernel_size=7, strides=2, padding="same")
        self.maxpool = layers.MaxPooling2D((2, 2), strides=2)
        
        # Single BottleneckLayer instance
        self.bottleneck = BottleneckLayer(growthRate)

    def call(self, x):
        # Initial conv + relu + maxpool
        y = self.maxpool(self.relu(self.conv1(x)))  # Shape approx (batch, H/4, W/4, 2*growthRate)
        
        # Applying 6 bottleneck layers
        # According to the issue, reusing the same bottleneck instance causes errors,
        # because the channel dimension grows after concat each time, but the layer's
        # conv expects fixed input channels.
        # So here, to reproduce the intended behavior, instantiate a new BottleneckLayer each loop.
        for _ in range(6):
            # Instantiate BottleneckLayer on the fly to handle changed channel dimension correctly
            y = BottleneckLayer(12)(y)
        
        return y

def my_model_function():
    # Return a new instance of MyModel
    return MyModel()

def GetInput():
    # Return a batch of random inputs with shape expected by the model:
    # The first conv is 7x7 stride 2, so input size can be e.g. 224x224 RGB images
    # Batch size 16 chosen based on original example prints
    return tf.random.uniform((16, 224, 224, 3), dtype=tf.float32)

