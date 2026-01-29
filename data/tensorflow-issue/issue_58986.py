# tf.random.uniform((B, 256, 256, 3), dtype=tf.uint8)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Normalization layer: scale inputs from [0,255] uint8 to [0,1] float32
        self.normalization = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)
        
        # Encoder blocks - contraction path
        self.conv1a = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.drop1 = tf.keras.layers.Dropout(0.1)
        self.conv1b = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D((2,2))

        self.conv2a = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.drop2 = tf.keras.layers.Dropout(0.1)
        self.conv2b = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D((2,2))

        self.conv3a = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.drop3 = tf.keras.layers.Dropout(0.2)
        self.conv3b = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D((2,2))

        self.conv4a = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.drop4 = tf.keras.layers.Dropout(0.2)
        self.conv4b = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.pool4 = tf.keras.layers.MaxPooling2D((2,2))

        self.conv5a = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.drop5 = tf.keras.layers.Dropout(0.3)
        self.conv5b = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')

        # Decoder blocks - expansive path with transposed convs and concatenations
        self.up6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')
        self.conv6a = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.drop6 = tf.keras.layers.Dropout(0.2)
        self.conv6b = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.up7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')
        self.conv7a = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.drop7 = tf.keras.layers.Dropout(0.2)
        self.conv7b = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.up8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')
        self.conv8a = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.drop8 = tf.keras.layers.Dropout(0.1)
        self.conv8b = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')

        self.up9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')
        self.conv9a = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')
        self.drop9 = tf.keras.layers.Dropout(0.1)
        self.conv9b = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')

        # Output: sigmoid activation for binary segmentation mask
        self.out_conv = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')

    def call(self, inputs, training=False):
        # Normalize input pixels to [0,1]
        x = self.normalization(inputs)

        # Encoder path
        c1 = self.conv1a(x)
        c1 = self.drop1(c1, training=training)
        c1 = self.conv1b(c1)
        p1 = self.pool1(c1)

        c2 = self.conv2a(p1)
        c2 = self.drop2(c2, training=training)
        c2 = self.conv2b(c2)
        p2 = self.pool2(c2)

        c3 = self.conv3a(p2)
        c3 = self.drop3(c3, training=training)
        c3 = self.conv3b(c3)
        p3 = self.pool3(c3)

        c4 = self.conv4a(p3)
        c4 = self.drop4(c4, training=training)
        c4 = self.conv4b(c4)
        p4 = self.pool4(c4)

        c5 = self.conv5a(p4)
        c5 = self.drop5(c5, training=training)
        c5 = self.conv5b(c5)

        # Decoder path with concatenation from encoder skip connections
        u6 = self.up6(c5)
        u6 = tf.concat([u6, c4], axis=-1)
        c6 = self.conv6a(u6)
        c6 = self.drop6(c6, training=training)
        c6 = self.conv6b(c6)

        u7 = self.up7(c6)
        u7 = tf.concat([u7, c3], axis=-1)
        c7 = self.conv7a(u7)
        c7 = self.drop7(c7, training=training)
        c7 = self.conv7b(c7)

        u8 = self.up8(c7)
        u8 = tf.concat([u8, c2], axis=-1)
        c8 = self.conv8a(u8)
        c8 = self.drop8(c8, training=training)
        c8 = self.conv8b(c8)

        u9 = self.up9(c8)
        u9 = tf.concat([u9, c1], axis=-1)
        c9 = self.conv9a(u9)
        c9 = self.drop9(c9, training=training)
        c9 = self.conv9b(c9)

        outputs = self.out_conv(c9)
        return outputs

def my_model_function():
    """
    Returns an instance of the MyModel class,
    compiled with Adam optimizer and binary crossentropy loss.
    """
    model = MyModel()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Compile model with binary crossentropy loss and accuracy metric
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def GetInput():
    """
    Returns a random uint8 tensor of shape (1, 256, 256, 3) matching the expected input of MyModel.
    Values are in [0,255].
    """
    return tf.random.uniform((1, 256, 256, 3), minval=0, maxval=256, dtype=tf.uint8)

