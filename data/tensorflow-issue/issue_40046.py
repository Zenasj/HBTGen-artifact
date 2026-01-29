# tf.random.uniform((B, 300, 300, 3), dtype=tf.float32) ← Input shape inferred from dataset and comments

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Fixing the tuple issue by removing trailing commas after layer definitions

        self.padding_1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))  # padding before conv1_1
        self.conv1_1 = tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', strides=1, activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', strides=1, activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.conv2_1 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', strides=1, activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', strides=1, activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.conv3_1 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', strides=1, activation='relu')
        self.conv3_2 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', strides=1, activation='relu')
        self.conv3_3 = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', strides=1, activation='relu')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.conv4_1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1, activation='relu')
        self.conv4_2 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1, activation='relu')
        self.conv4_3 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1, activation='relu')
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.conv5_1 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1, activation='relu')
        self.conv5_2 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1, activation='relu')
        self.conv5_3 = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', strides=1, activation='relu')
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.padding6 = tf.keras.layers.ZeroPadding2D(padding=(6, 6))  # padding before atrous convolution
        self.conv6 = tf.keras.layers.Conv2D(1024, kernel_size=3, padding='same',
                                            dilation_rate=6, activation='relu')  # atrous convolution
        self.conv7 = tf.keras.layers.Conv2D(1024, kernel_size=1, activation='relu')

    def call(self, x):
        x = self.padding_1(x)
        x = self.conv1_1(x)  # (N, 300, 300, 3 -> N, 300, 300, 3)
        x = self.conv1_2(x)  # (N, 300, 300, 64)
        x = self.pool1(x)    # (N, 150, 150, 64)

        x = self.conv2_1(x)  # (N, 150, 150, 128)
        x = self.conv2_2(x)  # (N, 150, 150, 128)
        x = self.pool2(x)    # (N, 75, 75, 128)

        x = self.conv3_1(x)  # (N, 75, 75, 256)
        x = self.conv3_2(x)  # (N, 75, 75, 256)
        x = self.conv3_3(x)  # (N, 75, 75, 256)
        x = self.pool3(x)    # (N, 37 or 38, 37 or 38, 256) roughly (38,38) due to ceil_mode analog

        x = self.conv4_1(x)  # (N, 38, 38, 512)
        x = self.conv4_2(x)  # (N, 38, 38, 512)
        x = self.conv4_3(x)  # (N, 38, 38, 512)
        conv4_3_feats = x    # Save for output
        x = self.pool4(x)    # (N, 19, 19, 512)

        x = self.conv5_1(x)  # (N, 19, 19, 512)
        x = self.conv5_2(x)  # (N, 19, 19, 512)
        x = self.conv5_3(x)  # (N, 19, 19, 512)
        x = self.pool5(x)    # (N, 9 or 10, 9 or 10, 512) but likely 19,19 doesn't reduce dims per comment

        x = self.padding6(x) # Padding (6,6)
        x = self.conv6(x)    # Atrous convolution (N, 19, 19, 1024)
        x = self.conv7(x)    # (N, 19, 19, 1024)
        conv7_feats = x

        return conv4_3_feats, conv7_feats


def my_model_function():
    # Return a new instance of MyModel with default initialization
    return MyModel()


def GetInput():
    # Return a random tensor with batch size 1 and input shape (300, 300, 3)
    # dtype float32 as typical for images scaled 0-1 or standard input->conv2D
    return tf.random.uniform((1, 300, 300, 3), dtype=tf.float32)

