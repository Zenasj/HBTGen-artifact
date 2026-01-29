# tf.random.uniform((1, 640, 640, 3), dtype=tf.float32), assuming input images are 640x640 RGB

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Placeholder backbone mimicking ResNet101 features extraction at multiple stages:
        # C2, C3, C4, C5 feature maps usually correspond to different spatial downsampling levels.
        # For simplicity, create convolutional blocks to simulate this behavior.
        
        # Conv stem
        self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='conv1')
        self.bn_conv1 = tf.keras.layers.BatchNormalization(name='bn_conv1')
        self.relu = tf.keras.layers.Activation('relu')
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        # Simulated residual blocks for C2, C3, C4, C5 with increasing depth and downsampling
        # Using simple conv blocks here as a conceptual stand-in for full ResNet101 blocks

        def conv_block(filters, blocks, stride=1, name=None):
            layers = []
            # First block with downsampling if stride !=1
            layers.append(tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False, name=name+'_conv1'))
            layers.append(tf.keras.layers.BatchNormalization(name=name+'_bn1'))
            layers.append(tf.keras.layers.Activation('relu'))
            for i in range(1, blocks):
                layers.append(tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'{name}_conv{i+1}'))
                layers.append(tf.keras.layers.BatchNormalization(name=f'{name}_bn{i+1}'))
                layers.append(tf.keras.layers.Activation('relu'))
            return tf.keras.Sequential(layers, name=name)

        self.C2 = conv_block(256, 3, stride=1, name='C2')
        self.C3 = conv_block(512, 4, stride=2, name='C3')
        self.C4 = conv_block(1024, 23, stride=2, name='C4')  # Deeper block, shorter for simplicity
        self.C5 = conv_block(2048, 3, stride=2, name='C5')

    def call(self, inputs, training=False):
        # Forward pass similar to resnet_graph: input image -> conv1 -> bn -> relu -> maxpool -> C2..C5

        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.C2(x, training=training)
        c3 = self.C3(c2, training=training)
        c4 = self.C4(c3, training=training)
        c5 = self.C5(c4, training=training)

        # Return feature maps as tuple like Mask R-CNN expects
        return (c2, c3, c4, c5)

def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Return a random 640x640 RGB float32 tensor matching expected input
    # Batch size 1, height 640, width 640, channels 3
    return tf.random.uniform((1, 640, 640, 3), dtype=tf.float32)

