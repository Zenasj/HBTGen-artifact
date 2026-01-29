# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape is (None, None, 1) for height, width, channels (grayscale image)

import tensorflow as tf

def get_crop_shape(target, query):
    # The shape tensors might be dynamic, so use tf.shape to get actual dimensions at runtime.
    # This avoids NoneType subtraction errors seen in original code.
    target_shape = tf.shape(target)
    query_shape = tf.shape(query)
    height_diff = target_shape[1] - query_shape[1]
    width_diff = target_shape[2] - query_shape[2]
    # Assert height_diff and width_diff >= 0 for padding logic; else fallback to zero padding safely.
    # Compute padding on sides
    height_pad_1 = height_diff // 2
    height_pad_2 = height_diff - height_pad_1
    width_pad_1 = width_diff // 2
    width_pad_2 = width_diff - width_pad_1
    return (height_pad_1, height_pad_2), (width_pad_1, width_pad_2)

def getAct(x):
    # Use ReLU activation, as in original code.
    return tf.keras.layers.ReLU()(x)

class MyModel(tf.keras.Model):
    def __init__(self, kernelSize=3, drop_level=0.5, nChannels=1):
        super().__init__()
        self.kernelSize = kernelSize
        self.drop_level = drop_level
        self.nChannels = nChannels

        # Encoder layers
        self.conv1_1 = tf.keras.layers.Conv2D(64, kernelSize, padding='same', kernel_initializer='he_normal')
        self.conv1_2 = tf.keras.layers.Conv2D(64, kernelSize, padding='same', kernel_initializer='he_normal')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2_1 = tf.keras.layers.Conv2D(128, kernelSize, padding='same', kernel_initializer='he_normal')
        self.conv2_2 = tf.keras.layers.Conv2D(128, kernelSize, padding='same', kernel_initializer='he_normal')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv3_1 = tf.keras.layers.Conv2D(256, kernelSize, padding='same', kernel_initializer='he_normal')
        self.conv3_2 = tf.keras.layers.Conv2D(256, kernelSize, padding='same', kernel_initializer='he_normal')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv4_1 = tf.keras.layers.Conv2D(512, kernelSize, padding='same', kernel_initializer='he_normal')
        self.conv4_2 = tf.keras.layers.Conv2D(512, kernelSize, padding='same', kernel_initializer='he_normal')
        self.drop4 = tf.keras.layers.Dropout(drop_level)
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

        # Bottleneck layers
        self.conv5_1 = tf.keras.layers.Conv2D(1024, kernelSize, padding='same', kernel_initializer='he_normal')
        self.conv5_2 = tf.keras.layers.Conv2D(1024, kernelSize, padding='same', kernel_initializer='he_normal')
        self.drop5 = tf.keras.layers.Dropout(drop_level)

        # Decoder layers
        self.up6 = tf.keras.layers.Conv2DTranspose(512, kernelSize, strides=(2,2), padding='same', kernel_initializer='he_normal')
        self.conv6_1 = tf.keras.layers.Conv2D(512, kernelSize, padding='same', kernel_initializer='he_normal')
        self.conv6_2 = tf.keras.layers.Conv2D(512, kernelSize, padding='same', kernel_initializer='he_normal')

        self.up7 = tf.keras.layers.Conv2DTranspose(256, kernelSize, strides=(2,2), padding='same', kernel_initializer='he_normal')
        self.conv7_1 = tf.keras.layers.Conv2D(256, kernelSize, padding='same', kernel_initializer='he_normal')
        self.conv7_2 = tf.keras.layers.Conv2D(256, kernelSize, padding='same', kernel_initializer='he_normal')

        self.up8 = tf.keras.layers.Conv2DTranspose(128, kernelSize, strides=(2,2), padding='same', kernel_initializer='he_normal')
        self.conv8_1 = tf.keras.layers.Conv2D(128, kernelSize, padding='same', kernel_initializer='he_normal')
        self.conv8_2 = tf.keras.layers.Conv2D(128, kernelSize, padding='same', kernel_initializer='he_normal')

        self.up9 = tf.keras.layers.Conv2DTranspose(64, kernelSize, strides=(2,2), padding='same', kernel_initializer='he_normal')
        self.conv9_1 = tf.keras.layers.Conv2D(64, kernelSize, padding='same', kernel_initializer='he_normal')
        self.conv9_2 = tf.keras.layers.Conv2D(64, kernelSize, padding='same', kernel_initializer='he_normal')

        self.conv9_3 = tf.keras.layers.Conv2D(2, kernelSize, padding='same', kernel_initializer='he_normal')

        self.conv10 = tf.keras.layers.Conv2D(nChannels, 1, activation='sigmoid')

        self.dropout4 = tf.keras.layers.Dropout(drop_level)
        self.dropout5 = tf.keras.layers.Dropout(drop_level)

    def call(self, inputs, training=False):
        # Encoder path
        conv1 = self.conv1_1(inputs)
        conv1 = tf.nn.relu(conv1)
        conv1 = self.conv1_2(conv1)
        conv1 = tf.nn.relu(conv1)
        # pool1 = self.pool1(conv1)  # commented out dropout in original. Use conv1 directly -> pool1.
        pool1 = self.pool1(conv1)

        conv2 = self.conv2_1(pool1)
        conv2 = tf.nn.relu(conv2)
        conv2 = self.conv2_2(conv2)
        conv2 = tf.nn.relu(conv2)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3_1(pool2)
        conv3 = tf.nn.relu(conv3)
        conv3 = self.conv3_2(conv3)
        conv3 = tf.nn.relu(conv3)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4_1(pool3)
        conv4 = tf.nn.relu(conv4)
        conv4 = self.conv4_2(conv4)
        conv4 = tf.nn.relu(conv4)
        drop4 = self.dropout4(conv4, training=training)
        pool4 = self.pool4(drop4)

        # Bottleneck
        conv5 = self.conv5_1(pool4)
        conv5 = tf.nn.relu(conv5)
        conv5 = self.conv5_2(conv5)
        conv5 = tf.nn.relu(conv5)
        drop5 = self.dropout5(conv5, training=training)

        # Decoder path with cropping and padding for concatenation

        up6 = self.up6(drop5)
        ch, cw = get_crop_shape(drop4, up6)
        up6 = tf.pad(up6, [[0,0], [ch[0], ch[1]], [cw[0], cw[1]], [0,0]])  # ZeroPadding2D equivalent
        merge6 = tf.concat([drop4, up6], axis=3)

        conv6 = self.conv6_1(merge6)
        conv6 = tf.nn.relu(conv6)
        conv6 = self.conv6_2(conv6)
        conv6 = tf.nn.relu(conv6)

        up7 = self.up7(conv6)
        ch, cw = get_crop_shape(conv3, up7)
        up7 = tf.pad(up7, [[0,0], [ch[0], ch[1]], [cw[0], cw[1]], [0,0]])
        merge7 = tf.concat([conv3, up7], axis=3)

        conv7 = self.conv7_1(merge7)
        conv7 = tf.nn.relu(conv7)
        conv7 = self.conv7_2(conv7)
        conv7 = tf.nn.relu(conv7)

        up8 = self.up8(conv7)
        ch, cw = get_crop_shape(conv2, up8)
        up8 = tf.pad(up8, [[0,0], [ch[0], ch[1]], [cw[0], cw[1]], [0,0]])
        merge8 = tf.concat([conv2, up8], axis=3)

        conv8 = self.conv8_1(merge8)
        conv8 = tf.nn.relu(conv8)
        conv8 = self.conv8_2(conv8)
        conv8 = tf.nn.relu(conv8)

        up9 = self.up9(conv8)
        ch, cw = get_crop_shape(conv1, up9)
        up9 = tf.pad(up9, [[0,0], [ch[0], ch[1]], [cw[0], cw[1]], [0,0]])
        merge9 = tf.concat([conv1, up9], axis=3)

        conv9 = self.conv9_1(merge9)
        conv9 = tf.nn.relu(conv9)
        conv9 = self.conv9_2(conv9)
        conv9 = tf.nn.relu(conv9)
        conv9 = self.conv9_3(conv9)
        conv9 = tf.nn.relu(conv9)

        conv10 = self.conv10(conv9)

        return conv10

def my_model_function():
    # Return an instance of MyModel with defaults
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected input: batch size 1, height 256, width 256, 1 channel.
    # Height and width can be variable but use a concrete example here.
    # dtype float32 as standard for TF models
    return tf.random.uniform((1, 256, 256, 1), dtype=tf.float32)

