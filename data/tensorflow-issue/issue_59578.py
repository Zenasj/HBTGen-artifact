# tf.random.uniform((B, 128, 128, 3), dtype=tf.float32)  # Assuming channel last, RGB images of 128x128 for segmentation

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder
        # Block 1
        self.conv1_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn1_1 = layers.BatchNormalization()
        self.conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn1_2 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))

        # Block 2
        self.conv2_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn2_1 = layers.BatchNormalization()
        self.conv2_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn2_2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))

        # Block 3
        self.conv3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.bn3_1 = layers.BatchNormalization()
        self.conv3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.bn3_2 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D((2, 2))

        # Block 4
        self.conv4_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn4_1 = layers.BatchNormalization()
        self.conv4_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn4_2 = layers.BatchNormalization()
        self.drop4 = layers.Dropout(0.5)
        self.pool4 = layers.MaxPooling2D((2, 2))

        # Block 5 (Bottleneck)
        self.conv5_1 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')
        self.bn5_1 = layers.BatchNormalization()
        self.conv5_2 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')
        self.bn5_2 = layers.BatchNormalization()
        self.drop5 = layers.Dropout(0.5)

        # Decoder
        self.up6 = layers.UpSampling2D((2, 2))
        self.conv6_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn6_1 = layers.BatchNormalization()
        self.conv6_2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn6_2 = layers.BatchNormalization()

        self.up7 = layers.UpSampling2D((2, 2))
        self.conv7_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.bn7_1 = layers.BatchNormalization()
        self.conv7_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.bn7_2 = layers.BatchNormalization()

        self.up8 = layers.UpSampling2D((2, 2))
        self.conv8_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn8_1 = layers.BatchNormalization()
        self.conv8_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn8_2 = layers.BatchNormalization()

        self.up9 = layers.UpSampling2D((2, 2))
        self.conv9_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn9_1 = layers.BatchNormalization()
        self.conv9_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn9_2 = layers.BatchNormalization()

        # Output layer
        self.output_conv = layers.Conv2D(1, (1, 1), activation='softmax')  # 1 class with softmax per channel

    def call(self, inputs, training=False):
        # Encoder
        x1 = self.conv1_1(inputs)
        x1 = self.bn1_1(x1, training=training)
        x1 = self.conv1_2(x1)
        x1 = self.bn1_2(x1, training=training)
        p1 = self.pool1(x1)

        x2 = self.conv2_1(p1)
        x2 = self.bn2_1(x2, training=training)
        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2, training=training)
        p2 = self.pool2(x2)

        x3 = self.conv3_1(p2)
        x3 = self.bn3_1(x3, training=training)
        x3 = self.conv3_2(x3)
        x3 = self.bn3_2(x3, training=training)
        p3 = self.pool3(x3)

        x4 = self.conv4_1(p3)
        x4 = self.bn4_1(x4, training=training)
        x4 = self.conv4_2(x4)
        x4 = self.bn4_2(x4, training=training)
        d4 = self.drop4(x4, training=training)
        p4 = self.pool4(d4)

        x5 = self.conv5_1(p4)
        x5 = self.bn5_1(x5, training=training)
        x5 = self.conv5_2(x5)
        x5 = self.bn5_2(x5, training=training)
        d5 = self.drop5(x5, training=training)

        # Decoder with skip connections
        up6 = self.up6(d5)
        concat6 = tf.concat([up6, d4], axis=-1)
        c6 = self.conv6_1(concat6)
        c6 = self.bn6_1(c6, training=training)
        c6 = self.conv6_2(c6)
        c6 = self.bn6_2(c6, training=training)

        up7 = self.up7(c6)
        concat7 = tf.concat([up7, x3], axis=-1)
        c7 = self.conv7_1(concat7)
        c7 = self.bn7_1(c7, training=training)
        c7 = self.conv7_2(c7)
        c7 = self.bn7_2(c7, training=training)

        up8 = self.up8(c7)
        concat8 = tf.concat([up8, x2], axis=-1)
        c8 = self.conv8_1(concat8)
        c8 = self.bn8_1(c8, training=training)
        c8 = self.conv8_2(c8)
        c8 = self.bn8_2(c8, training=training)

        up9 = self.up9(c8)
        concat9 = tf.concat([up9, x1], axis=-1)
        c9 = self.conv9_1(concat9)
        c9 = self.bn9_1(c9, training=training)
        c9 = self.conv9_2(c9)
        c9 = self.bn9_2(c9, training=training)

        outputs = self.output_conv(c9)
        return outputs


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assume batch size 1 for the random input and float32 dtype
    return tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)

