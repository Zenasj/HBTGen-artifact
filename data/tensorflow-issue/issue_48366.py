# tf.random.uniform((B, 512, 512, 3), dtype=tf.float32) ← Input tensor shape and type inferred from CFG target size and 3 channels

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        initializer = 'he_normal'

        # Encoder Block 1
        self.conv_enc_1_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_enc_1_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)

        # Encoder Block 2
        self.max_pool_enc_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv_enc_2_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_enc_2_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)

        # Encoder Block 3
        self.max_pool_enc_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv_enc_3_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_enc_3_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)

        # Bottleneck
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv_bottle_1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_bottle_2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)

        # Decoder Block 1
        self.up_dec_1_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.up_dec_1_conv = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_dec_1_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_dec_1_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)

        # Decoder Block 2
        self.up_dec_2_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.up_dec_2_conv = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_dec_2_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_dec_2_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)

        # Decoder Block 3
        self.up_dec_3_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.up_dec_3_conv = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_dec_3_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_dec_3_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)
        self.conv_dec_3_3 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=initializer)

        # Final output layer
        self.output_conv = tf.keras.layers.Conv2D(self.cfg["n_classes"], 1, activation='softmax')

    def call(self, inputs, training=False):
        # Encoder
        x = self.conv_enc_1_1(inputs)
        conv_enc_1 = self.conv_enc_1_2(x)

        x = self.max_pool_enc_2(conv_enc_1)
        x = self.conv_enc_2_1(x)
        conv_enc_2 = self.conv_enc_2_2(x)

        x = self.max_pool_enc_3(conv_enc_2)
        x = self.conv_enc_3_1(x)
        conv_enc_3 = self.conv_enc_3_2(x)

        # Bottleneck
        x = self.max_pool(conv_enc_3)
        x = self.conv_bottle_1(x)
        x = self.conv_bottle_2(x)

        # Decoder
        x = self.up_dec_1_upsample(x)
        x = self.up_dec_1_conv(x)
        x = tf.keras.layers.concatenate([conv_enc_3, x], axis=3)
        x = self.conv_dec_1_1(x)
        x = self.conv_dec_1_2(x)

        x = self.up_dec_2_upsample(x)
        x = self.up_dec_2_conv(x)
        x = tf.keras.layers.concatenate([conv_enc_2, x], axis=3)
        x = self.conv_dec_2_1(x)
        x = self.conv_dec_2_2(x)

        x = self.up_dec_3_upsample(x)
        x = self.up_dec_3_conv(x)
        x = tf.keras.layers.concatenate([conv_enc_1, x], axis=3)
        x = self.conv_dec_3_1(x)
        x = self.conv_dec_3_2(x)
        x = self.conv_dec_3_3(x)

        output = self.output_conv(x)
        return output

def my_model_function():
    # Configuration dictionary as per original issue
    CFG = dict(
        batch_size=8,
        n_classes=15,
        epochs=100,
        target_height=512,
        target_width=512,
        n_channels=3,
        replace_with_label=15
    )
    model = MyModel(CFG)
    return model

def GetInput():
    # Return a random tensor input with batch size = batch_size, shape (height, width, channels)
    B = 8  # batch size
    H = 512  # height per CFG target_height
    W = 512  # width per CFG target_width
    C = 3  # channels per config
    # Generate float32 tensor with values between 0 and 1 (as images normalized in original code)
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)

