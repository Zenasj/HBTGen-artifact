# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)  ← CIFAR-100 input shape inferred from original code

import tensorflow as tf
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten

# Note: The original issue uses keras 2.2.4 with TF 1.x, here we adapt to TF 2.20.0 compatible code.


class MyModel(tf.keras.Model):
    def __init__(self, num_classes=100, input_shape=(32, 32, 3)):
        super().__init__()
        # CIFAR100 CNN model as defined in the issue
        self.conv1a = Conv2D(128, (3, 3), padding='same', strides=1, input_shape=input_shape)
        self.act1a = Activation('relu')
        self.conv1b = Conv2D(128, (3, 3))
        self.act1b = Activation('relu')
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
        self.drop1 = Dropout(0.1)
        
        self.conv2a = Conv2D(256, (3, 3), padding='same', strides=1)
        self.act2a = Activation('relu')
        self.conv2b = Conv2D(256, (3, 3))
        self.act2b = Activation('relu')
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
        self.drop2 = Dropout(0.25)
        
        self.conv3a = Conv2D(512, (3, 3), padding='same', strides=1)
        self.act3a = Activation('relu')
        self.conv3b = Conv2D(512, (3, 3))
        self.act3b = Activation('relu')
        self.pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
        self.drop3 = Dropout(0.5)
        
        self.flatten = Flatten()
        self.fc1 = Dense(1024)
        self.actfc1 = Activation('relu')
        self.dropfc1 = Dropout(0.5)
        self.fc_out = Dense(num_classes)
        self.act_out = Activation('softmax')

    def call(self, inputs, training=False):
        x = self.conv1a(inputs)
        x = self.act1a(x)
        x = self.conv1b(x)
        x = self.act1b(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        x = self.conv2a(x)
        x = self.act2a(x)
        x = self.conv2b(x)
        x = self.act2b(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        x = self.conv3a(x)
        x = self.act3a(x)
        x = self.conv3b(x)
        x = self.act3b(x)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.actfc1(x)
        x = self.dropfc1(x, training=training)
        x = self.fc_out(x)
        x = self.act_out(x)
        return x


def my_model_function():
    # Use the commonly expected parameters from the issue:
    # num_classes=100 (CIFAR-100), input_shape=(32,32,3)
    return MyModel(num_classes=100, input_shape=(32, 32, 3))


def GetInput():
    # Generate a batch of inputs with shape [batch_size, 32, 32, 3]
    # Using batch size of 8 as a reasonable example
    batch_size = 8
    # Input dtype float32 and pixel range normalized to [0,1], inferred from NormalizeData dividing by 255.
    return tf.random.uniform((batch_size, 32, 32, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

