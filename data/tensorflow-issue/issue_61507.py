# tf.random.uniform((64, 28, 28, 1), dtype=tf.float32) ← batch_size=64, image size 28x28 grayscale

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Model, Sequential

class ResNetBlock(Model):
    def __init__(self, filters=64, strides=1):
        super(ResNetBlock, self).__init__()
        self.strides = strides
        self.c1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')
        self.b2 = BatchNormalization()

        if strides > 1:
            self.c3 = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')
            self.b3 = BatchNormalization()
        else:
            self.c3 = None
            self.b3 = None

        self.a2 = Activation('relu')

    def call(self, inputs, training=False):
        short_x = inputs
        x = self.c1(inputs)
        x = self.b1(x, training=training)
        x = self.a1(x)
        x = self.c2(x)
        y = self.b2(x, training=training)
        if self.strides > 1:
            short_x = self.c3(short_x)
            short_x = self.b3(short_x, training=training)
        return self.a2(short_x + y)

class MyModel(Model):
    def __init__(self, model_lst=[2, 2, 2, 2], cur_filters=64):
        super(MyModel, self).__init__()
        self.c1 = Conv2D(filters=cur_filters, kernel_size=(7, 7), strides=2, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.blocks = Sequential()
        current_filters = cur_filters
        for i, blocks_num in enumerate(model_lst):
            for idx in range(blocks_num):
                if i != 0 and idx == 0:
                    block = ResNetBlock(current_filters, strides=2)
                else:
                    block = ResNetBlock(current_filters, strides=1)
                self.blocks.add(block)
            current_filters *= 2

        self.g1 = GlobalAveragePooling2D()
        # 10 output classes with softmax activation and L2 regularization
        self.d1 = Dense(10, activation='softmax', kernel_regularizer=L2())

    def call(self, inputs, training=False):
        x = self.c1(inputs)
        x = self.b1(x, training=training)
        x = self.a1(x)
        x = self.p1(x)

        x = self.blocks(x, training=training)
        x = self.g1(x)
        y = self.d1(x)
        return y

def my_model_function():
    # Return an instance of the ResNet18-like model
    # Based on model_lst=[2,2,2,2] blocks per stage as per standard ResNet-18
    model = MyModel([2, 2, 2, 2], cur_filters=64)
    return model

def GetInput():
    # Return a random input tensor matching the model input shape: batch_size=64, 28x28 grayscale images
    # Values are scaled between 0 and 1 similar to normalized fashion_mnist data
    batch_size = 64
    height = 28
    width = 28
    channels = 1
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

# The model is compatible with TensorFlow 2.20.0 and suitable for tf.function + XLA compilation.
# Example usage (commented out):
#
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

