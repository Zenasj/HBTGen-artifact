import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model.save(MODEL_DIR, save_format='tf')

tf.keras.models.load_model(MODEL_DIR)

class BottleNeck(tf.keras.layers.Layer):  
    expansion = 4 

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(1, 1), use_bias=False)  
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(3, 3), strides=(stride, stride), use_bias=False) 
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=out_channels * BottleNeck.expansion, kernel_size=(1, 1), use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.downsample = tf.keras.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:            
            self.downsample.add(tf.keras.layers.Conv2D(filters=out_channels * BottleNeck.expansion, kernel_size=(1, 1), strides=(stride, stride), use_bias=False))
            self.downsample.add(tf.keras.layers.BatchNormalization())
  
    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = tf.nn.relu(out)

        out = self.pad1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = tf.nn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        down = self.downsample(inputs)
        out += down
        out = tf.nn.relu(out)

        return out

    
class ResNet(tf.keras.Model):
    def __init__(self, dataset, block, num_blocks, num_classes):
        super(ResNet, self).__init__()        
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.in_channels = 64
            self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
            self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), use_bias=False)  
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.relu = tf.nn.relu

            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
            self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
            self.fc = tf.keras.layers.Dense(num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1, training=None):     
        strides = [stride] + [1] * (num_blocks - 1)
        layers = tf.keras.Sequential()        
        for stride in strides:
            layers.add(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            
        return layers

    def call(self, x, training=None):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            
            x = self.pad1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x, training=training)
            x = self.layer2(x, training=training)
            x = self.layer3(x, training=training)
            x = self.layer4(x, training=training)
            x = self.avgpool(x)
            x = tf.keras.layers.Flatten()(x)
            x = self.fc(x)

        return x