# tf.random.uniform((B, 299, 299, 1), dtype=tf.float32)  # Input shape inferred from data loading and preprocessing: grayscale image resized to (299,299) with channel=1

import tensorflow as tf

class ConvBnAct(tf.keras.layers.Layer):
    def __init__(
        self,
        filters=64,
        kernel_size=(3,3),
        activation='relu',
        padding='same',
        name='conv'):
        super().__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding,
            name=name+'_conv')
        self.norm = tf.keras.layers.BatchNormalization(name=name+'_bn')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.norm(x, training=training)
        return x

class Block_1(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Block_1")
        self.conv1 = ConvBnAct(64, name='block1_conv1')
        self.conv2 = ConvBnAct(64, name='block1_conv2')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.pool(x)
        return x

class Block_2(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Block_2")
        self.conv1 = ConvBnAct(128, name='block2_conv1')
        self.conv2 = ConvBnAct(128, name='block2_conv2')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.pool(x)
        return x

class Block_3(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Block_3")
        self.conv1 = ConvBnAct(256, name='block3_conv1')
        self.conv2 = ConvBnAct(256, name='block3_conv2')
        self.conv3 = ConvBnAct(256, name='block3_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool(x)
        return x

class Block_4(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Block_4")
        self.conv1 = ConvBnAct(512, name='block4_conv1')
        self.conv2 = ConvBnAct(512, name='block4_conv2')
        self.conv3 = ConvBnAct(512, name='block4_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool(x)
        return x

class Block_5(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Block_5")
        self.conv1 = ConvBnAct(512, name='block5_conv1')
        self.conv2 = ConvBnAct(512, name='block5_conv2')
        self.conv3 = ConvBnAct(512, name='block5_conv3')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool(x)
        return x

class MyModel(tf.keras.Model):
    """
    Reimplementation/fusion of the VGG_PR model as the main model.

    This model uses Conv2D + BatchNormalization + ReLU blocks grouped into VGG-style blocks,
    ending with global average pooling and dense layers.

    Assumptions:
      - num_classes = 20 by default as per the original code.
      - Inputs expected are grayscale images of shape (299,299,1).
      - The training argument is passed appropriately to BatchNorm layers.
    """
    def __init__(self, num_classes=20):
        super(MyModel, self).__init__()
        self.block1 = Block_1()
        self.block2 = Block_2()
        self.block3 = Block_3()
        self.block4 = Block_4()
        self.block5 = Block_5()
        self.avg = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', name='fc1')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', name='fc2')
        self.fc3 = tf.keras.layers.Dense(num_classes, activation='linear', name='predictions')

    def call(self, inputs, training=False):
        x = self.block1(inputs, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.avg(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default number of classes (20).
    return MyModel(num_classes=20)

def GetInput():
    # Create a random tensor with shape (batch_size, 299, 299, 1), dtype float32.
    # Matches the input shape of the model as inferred from the data pipeline.
    batch_size = 8  # common batch size from original code
    input_shape = (batch_size, 299, 299, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)

