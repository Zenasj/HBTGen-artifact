# tf.random.uniform((B, 512, 512, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Hyperparameters and configurations
        self.ks = 3
        self.rg = 0.01
        self.activation = 'relu'
        self.ch1 = 32
        self.ch2 = self.ch1 * 2
        self.ch3 = self.ch2 * 2
        self.ch4 = self.ch3 * 2
        self.ch5 = self.ch4 * 2
        self.init = tf.keras.initializers.glorot_normal(seed=0)
        # Layers for encoder
        self.conv_in = tf.keras.layers.Conv2D(self.ch1, self.ks, padding='same', use_bias=True,
                                              kernel_initializer=self.init,
                                              kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                              bias_initializer='zeros')
        self.conv_x1_1 = tf.keras.layers.Conv2D(self.ch1, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.act_x1 = tf.keras.layers.Activation(self.activation)
        self.bn_x1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.add_x1 = tf.keras.layers.Add()
        self.pool_x1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')

        self.conv_x2_0 = tf.keras.layers.Conv2D(self.ch2, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv_x2_1 = tf.keras.layers.Conv2D(self.ch2, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.act_x2 = tf.keras.layers.Activation(self.activation)
        self.bn_x2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.add_x2 = tf.keras.layers.Add()
        self.pool_x2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')

        self.conv_x3_0 = tf.keras.layers.Conv2D(self.ch3, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv_x3_1 = tf.keras.layers.Conv2D(self.ch3, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.act_x3 = tf.keras.layers.Activation(self.activation)
        self.bn_x3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.add_x3 = tf.keras.layers.Add()
        self.pool_x3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')

        self.conv_x4_0 = tf.keras.layers.Conv2D(self.ch4, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv_x4_1 = tf.keras.layers.Conv2D(self.ch4, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.act_x4 = tf.keras.layers.Activation(self.activation)
        self.bn_x4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.add_x4 = tf.keras.layers.Add()
        self.pool_x4 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')

        self.conv_x5_0 = tf.keras.layers.Conv2D(self.ch5, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv_x5_1 = tf.keras.layers.Conv2D(self.ch5, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.act_x5 = tf.keras.layers.Activation(self.activation)
        self.bn_x5 = tf.keras.layers.BatchNormalization(axis=-1)
        self.add_x5 = tf.keras.layers.Add()
        self.pool_x5 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')  # encoded output size: 16x16

        # "xu" conv 1x1 after encoding
        self.conv_xu = tf.keras.layers.Conv2D(self.ch5, (1, 1), padding='valid')

        # Decoder layers
        self.up6 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.conv6_1x1 = tf.keras.layers.Conv2D(self.ch5, 1, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv6_ks1 = tf.keras.layers.Conv2D(self.ch5, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv6_ks2 = tf.keras.layers.Conv2D(self.ch5, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.act6 = tf.keras.layers.Activation(self.activation)
        self.bn6 = tf.keras.layers.BatchNormalization(axis=-1)

        self.up7 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.conv7_1x1 = tf.keras.layers.Conv2D(self.ch4, 1, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv7_ks1 = tf.keras.layers.Conv2D(self.ch4, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv7_ks2 = tf.keras.layers.Conv2D(self.ch4, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.act7 = tf.keras.layers.Activation(self.activation)
        self.bn7 = tf.keras.layers.BatchNormalization(axis=-1)

        self.up8 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.conv8_1x1 = tf.keras.layers.Conv2D(self.ch3, 1, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv8_ks1 = tf.keras.layers.Conv2D(self.ch3, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv8_ks2 = tf.keras.layers.Conv2D(self.ch3, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.act8 = tf.keras.layers.Activation(self.activation)
        self.bn8 = tf.keras.layers.BatchNormalization(axis=-1)

        self.up9 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.conv9_1x1 = tf.keras.layers.Conv2D(self.ch2, 1, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.conv9_ks1 = tf.keras.layers.Conv2D(self.ch2, self.ks, padding='same', use_bias=True,
                                                kernel_initializer=self.init,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                bias_initializer='zeros')
        self.act9 = tf.keras.layers.Activation(self.activation)
        self.bn9 = tf.keras.layers.BatchNormalization(axis=-1)

        self.up10 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.conv10_1x1 = tf.keras.layers.Conv2D(self.ch1, 1, padding='same', use_bias=True,
                                                 kernel_initializer=self.init,
                                                 kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                 bias_initializer='zeros')
        self.conv10_ks1 = tf.keras.layers.Conv2D(self.ch1, self.ks, padding='same', use_bias=True,
                                                 kernel_initializer=self.init,
                                                 kernel_regularizer=tf.keras.regularizers.l2(self.rg),
                                                 bias_initializer='zeros')
        self.act10 = tf.keras.layers.Activation(self.activation)
        self.bn10 = tf.keras.layers.BatchNormalization(axis=-1)

        # Final conv and softmax activation for 2 classes
        self.final_conv = tf.keras.layers.Conv2D(2, (1, 1), padding='valid', use_bias=True)
        self.final_softmax = tf.keras.layers.Activation('softmax')

        # Concatenate layer to be reused
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs, training=False):
        # Encoder path
        xin = self.conv_in(inputs)
        x1 = self.conv_x1_1(xin)
        x1 = self.act_x1(x1)
        x1 = self.bn_x1(x1, training=training)
        x1 = self.add_x1([xin, x1])
        x1_pool = self.pool_x1(x1)  # 256 x 256

        xin = self.conv_x2_0(x1_pool)
        x2 = self.conv_x2_1(xin)
        x2 = self.act_x2(x2)
        x2 = self.bn_x2(x2, training=training)
        x2 = self.add_x2([xin, x2])
        x2_pool = self.pool_x2(x2)  # 128 x 128

        xin = self.conv_x3_0(x2_pool)
        x3 = self.conv_x3_1(xin)
        x3 = self.act_x3(x3)
        x3 = self.bn_x3(x3, training=training)
        x3 = self.add_x3([xin, x3])
        x3_pool = self.pool_x3(x3)  # 64 x 64

        xin = self.conv_x4_0(x3_pool)
        x4 = self.conv_x4_1(xin)
        x4 = self.act_x4(x4)
        x4 = self.bn_x4(x4, training=training)
        x4 = self.add_x4([xin, x4])
        x4_pool = self.pool_x4(x4)  # 32 x 32

        xin = self.conv_x5_0(x4_pool)
        x5 = self.conv_x5_1(xin)
        x5 = self.act_x5(x5)
        x5 = self.bn_x5(x5, training=training)
        x5 = self.add_x5([xin, x5])
        encoded = self.pool_x5(x5)  # 16 x 16

        # Decoder path
        xu = self.conv_xu(encoded)

        x6 = self.concat([xu, encoded])  # 16x16 concatenation
        x6 = self.up6(x6)  # 32x32
        x6 = self.conv6_1x1(x6)
        x6 = self.conv6_ks1(x6)
        x6 = self.conv6_ks2(x6)
        x6 = self.act6(x6)
        x6 = self.bn6(x6, training=training)

        x7 = self.concat([x6, x4])
        x7 = self.up7(x7)  # 64x64
        x7 = self.conv7_1x1(x7)
        x7 = self.conv7_ks1(x7)
        x7 = self.conv7_ks2(x7)
        x7 = self.act7(x7)
        x7 = self.bn7(x7, training=training)

        x8 = self.concat([x7, x3])
        x8 = self.up8(x8)  # 128x128
        x8 = self.conv8_1x1(x8)
        x8 = self.conv8_ks1(x8)
        x8 = self.conv8_ks2(x8)
        x8 = self.act8(x8)
        x8 = self.bn8(x8, training=training)

        x9 = self.concat([x8, x2])
        x9 = self.up9(x9)  # 256x256
        x9 = self.conv9_1x1(x9)
        x9 = self.conv9_ks1(x9)
        x9 = self.act9(x9)
        x9 = self.bn9(x9, training=training)

        x10 = self.concat([x9, x1])
        x10 = self.up10(x10)  # 512x512
        x10 = self.conv10_1x1(x10)
        x10 = self.conv10_ks1(x10)
        x10 = self.act10(x10)
        x10 = self.bn10(x10, training=training)

        decoded = self.final_conv(x10)
        d1 = self.final_softmax(decoded)

        return d1


def my_model_function():
    # Returns an instance of MyModel
    model = MyModel()
    # Compile the model with same optimizer, loss and metric as described
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def GetInput():
    # Provides a random tensor matching input shape (batch size 4 chosen arbitrarily)
    # dtype float32 typical for images
    batch_size = 4
    input_shape = (batch_size, 512, 512, 1)
    return tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)

