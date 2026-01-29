# tf.random.uniform((8, 128, 128, 3), dtype=tf.float32)  # Assuming batch size 8 based on batch_size in original code

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout,
    UpSampling2D, concatenate, Softmax
)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Input shape is (128, 128, 3)
        self.n_classes = 1  # per original code, number of segmentation classes
        
        # Encoder
        self.conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn1_1 = BatchNormalization()
        self.conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn1_2 = BatchNormalization()
        self.pool1 = MaxPooling2D((2, 2))

        self.conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn2_1 = BatchNormalization()
        self.conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn2_2 = BatchNormalization()
        self.pool2 = MaxPooling2D((2, 2))

        self.conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.bn3_1 = BatchNormalization()
        self.conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.bn3_2 = BatchNormalization()
        self.pool3 = MaxPooling2D((2, 2))

        self.conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn4_1 = BatchNormalization()
        self.conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn4_2 = BatchNormalization()
        self.drop4 = Dropout(0.5)
        self.pool4 = MaxPooling2D((2, 2))

        self.conv5_1 = Conv2D(1024, (3, 3), activation='relu', padding='same')
        self.bn5_1 = BatchNormalization()
        self.conv5_2 = Conv2D(1024, (3, 3), activation='relu', padding='same')
        self.bn5_2 = BatchNormalization()
        self.drop5 = Dropout(0.5)

        # Decoder
        self.up6 = UpSampling2D((2, 2))
        self.conv6_1 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn6_1 = BatchNormalization()
        self.conv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn6_2 = BatchNormalization()

        self.up7 = UpSampling2D((2, 2))
        self.conv7_1 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.bn7_1 = BatchNormalization()
        self.conv7_2 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.bn7_2 = BatchNormalization()

        self.up8 = UpSampling2D((2, 2))
        self.conv8_1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn8_1 = BatchNormalization()
        self.conv8_2 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn8_2 = BatchNormalization()

        self.up9 = UpSampling2D((2, 2))
        self.conv9_1 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn9_1 = BatchNormalization()
        self.conv9_2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn9_2 = BatchNormalization()

        # Final output layer; note that original code uses activation='softmax' on output with n_classes=1,
        # which is unusual (softmax with 1 class == 1),
        # but we replicate that as closely as possible.
        self.outputs_conv = Conv2D(self.n_classes, (1, 1), activation='softmax')

    def call(self, inputs, training=False):
        # Encoder path
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
        x4 = self.drop4(x4, training=training)
        p4 = self.pool4(x4)

        x5 = self.conv5_1(p4)
        x5 = self.bn5_1(x5, training=training)
        x5 = self.conv5_2(x5)
        x5 = self.bn5_2(x5, training=training)
        x5 = self.drop5(x5, training=training)
        
        # Decoder path with concat skip connections
        u6 = self.up6(x5)
        u6 = concatenate([u6, x4], axis=-1)
        x6 = self.conv6_1(u6)
        x6 = self.bn6_1(x6, training=training)
        x6 = self.conv6_2(x6)
        x6 = self.bn6_2(x6, training=training)

        u7 = self.up7(x6)
        u7 = concatenate([u7, x3], axis=-1)
        x7 = self.conv7_1(u7)
        x7 = self.bn7_1(x7, training=training)
        x7 = self.conv7_2(x7)
        x7 = self.bn7_2(x7, training=training)

        u8 = self.up8(x7)
        u8 = concatenate([u8, x2], axis=-1)
        x8 = self.conv8_1(u8)
        x8 = self.bn8_1(x8, training=training)
        x8 = self.conv8_2(x8)
        x8 = self.bn8_2(x8, training=training)

        u9 = self.up9(x8)
        u9 = concatenate([u9, x1], axis=-1)
        x9 = self.conv9_1(u9)
        x9 = self.bn9_1(x9, training=training)
        x9 = self.conv9_2(x9)
        x9 = self.bn9_2(x9, training=training)

        outputs = self.outputs_conv(x9)

        return outputs

def my_model_function():
    # Instantiate the model
    model = MyModel()
    # Compile with given settings from the original issue:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a valid random tensor input matching (batch_size=8, 128, 128, 3)
    batch_size = 8
    height = 128
    width = 128
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

