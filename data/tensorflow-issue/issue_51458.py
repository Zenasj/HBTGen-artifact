# tf.random.uniform((B, 512, 512, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Activation, Conv2D, Dropout,
    MaxPooling2D, Conv2DTranspose, concatenate, Lambda, Input
)
from tensorflow.keras import backend as K

IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initial Conv block
        self.lambda_norm = Lambda(lambda x: x / 1)  # identity normalization
        self.conv_c1 = Conv2D(48, (3, 3), activation=None, kernel_initializer='he_normal', padding='same')
        self.bn_c1 = BatchNormalization()
        self.act_c1 = Activation('relu')

        # Following dense blocks, transitions down and up as defined below
        # Growth rate for dense blocks
        self.growth_rate = 16

        # Dense blocks and transitions down
        # We define callable layers for each dense_block and transition layers because dense_block has logic
        # that can't directly map to a single layer; replicate that logic in call()
        
        # Store filters for transition downs (trans_down)
        self.trans_down_filters = [112, 192, 304, 464, 656]

        # Conv2D for transition_down 1x1 conv layers
        self.trans_down_convs = [
            Conv2D(filt, (1, 1), activation=None, kernel_initializer='he_normal', padding='same') 
            for filt in self.trans_down_filters
        ]
        self.trans_down_bns = [BatchNormalization() for _ in self.trans_down_filters]
        self.trans_down_acts = [Activation('relu') for _ in self.trans_down_filters]
        self.trans_down_dropouts = [Dropout(0.2) for _ in self.trans_down_filters]
        self.trans_down_pools = [MaxPooling2D((2, 2)) for _ in self.trans_down_filters]

        # Conv2DTranspose layers for trans_up
        self.trans_up_convs = [
            Conv2DTranspose(filt, (3,3), strides=(2,2), padding='same') 
            for filt in [240, 192, 160, 112, 80]
        ]

        # Final output conv
        self.output_conv = Conv2D(1, (1,1), activation='sigmoid')

        # BatchNorm/Activation and Dropout used inside dense_block
        self.bn_dense = BatchNormalization()
        self.act_dense = Activation('relu')
        self.dropout_dense = Dropout(0.2)

    def dense_block(self, inputs, growth_rate, n_layers):
        """
        Custom dense block as per original code.
        Returns tuple (dense_out, concatenated_features)
        dense_out is all outputs concatenated along channel axis
        concatenated_features is inputs concatenated with all intermediate outputs (used for next layer input)
        """
        total_features = []
        ini = inputs
        for i in range(n_layers):
            x = BatchNormalization()(ini)
            x = Activation('relu')(x)
            x = Conv2D(growth_rate, (3,3), activation=None, kernel_initializer='he_normal', padding='same')(x)
            x = Dropout(0.2)(x)
            total_features.append(x)
            ini = concatenate([x, ini], axis=3)
        # Concatenate all feature maps of this block
        dense_out = total_features[0]
        for j in range(len(total_features) - 1):
            dense_out = concatenate([dense_out, total_features[j+1]], axis=3)
        return dense_out, ini

    def trans_down(self, inputs, conv_layer, bn_layer, act_layer, dropout_layer, pool_layer):
        """
        Apply BatchNorm -> Relu -> Conv2D (1x1) -> Dropout -> MaxPool2D
        """
        x = bn_layer(inputs)
        x = act_layer(x)
        x = conv_layer(x)
        x = dropout_layer(x)
        x = pool_layer(x)
        return x

    def trans_up(self, inputs, conv_transpose_layer):
        """
        Conv2DTranspose with given filters, kernel 3x3, stride 2 and padding same for upsampling
        """
        return conv_transpose_layer(inputs)

    def call(self, inputs, training=None):
        s = self.lambda_norm(inputs)

        # Initial conv block c1
        c1 = self.conv_c1(s)
        c1 = self.bn_c1(c1, training=training)
        c1 = self.act_c1(c1)

        # Dense block 1 and transition down 1
        b1, _ = self.dense_block(c1, self.growth_rate, 4)
        con1 = concatenate([b1, c1], axis=3)
        d1 = self.trans_down(d1:=con1, *self.trans_down_convs[0:1], *self.trans_down_bns[0:1], *self.trans_down_acts[0:1], *self.trans_down_dropouts[0:1], *self.trans_down_pools[0:1])

        # Dense block 2 and trans down 2
        b2, _ = self.dense_block(d1, self.growth_rate, 5)
        con2 = concatenate([b2, d1], axis=3)
        d2 = self.trans_down(con2, *self.trans_down_convs[1:2], *self.trans_down_bns[1:2], *self.trans_down_acts[1:2], *self.trans_down_dropouts[1:2], *self.trans_down_pools[1:2])

        # Dense block 3 and trans down 3
        b3, _ = self.dense_block(d2, self.growth_rate, 7)
        con3 = concatenate([b3, d2], axis=3)
        d3 = self.trans_down(con3, *self.trans_down_convs[2:3], *self.trans_down_bns[2:3], *self.trans_down_acts[2:3], *self.trans_down_dropouts[2:3], *self.trans_down_pools[2:3])

        # Dense block 4 and trans down 4
        b4, _ = self.dense_block(d3, self.growth_rate, 10)
        con4 = concatenate([b4, d3], axis=3)
        d4 = self.trans_down(con4, *self.trans_down_convs[3:4], *self.trans_down_bns[3:4], *self.trans_down_acts[3:4], *self.trans_down_dropouts[3:4], *self.trans_down_pools[3:4])

        # Dense block 5 and trans down 5
        b5, _ = self.dense_block(d4, self.growth_rate, 12)
        con5 = concatenate([b5, d4], axis=3)
        d5 = self.trans_down(con5, *self.trans_down_convs[4:5], *self.trans_down_bns[4:5], *self.trans_down_acts[4:5], *self.trans_down_dropouts[4:5], *self.trans_down_pools[4:5])

        # Dense block 6 (bottom)
        b6, block_to_up6 = self.dense_block(d5, self.growth_rate, 15)

        # Upsampling paths
        u7 = self.trans_up(block_to_up6, self.trans_up_convs[0])
        con7 = concatenate([u7, con5], axis=3)
        b7, block_to_up7 = self.dense_block(con7, self.growth_rate, 12)

        u8 = self.trans_up(block_to_up7, self.trans_up_convs[1])
        con8 = concatenate([u8, con4], axis=3)
        b8, block_to_up8 = self.dense_block(con8, self.growth_rate, 10)

        u9 = self.trans_up(block_to_up8, self.trans_up_convs[2])
        con9 = concatenate([u9, con3], axis=3)
        b9, block_to_up9 = self.dense_block(con9, self.growth_rate, 7)

        u10 = self.trans_up(block_to_up9, self.trans_up_convs[3])
        con10 = concatenate([u10, con2], axis=3)
        b10, block_to_up10 = self.dense_block(con10, self.growth_rate, 5)

        u11 = self.trans_up(block_to_up10, self.trans_up_convs[4])
        con11 = concatenate([u11, con1], axis=3)
        b11, block_to_up11 = self.dense_block(con11, self.growth_rate, 4)

        outputs = self.output_conv(block_to_up11)
        return outputs

def my_model_function():
    model = MyModel()
    # Compile with RMSprop and dice coefficient lost as in original
    def dice_coeff(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-4),
                  loss='binary_crossentropy',
                  metrics=[dice_coeff])
    return model

def GetInput():
    # Return a random tensor input of shape (1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS) with float32 dtype,
    # normalized to [0, 1]
    x = tf.random.uniform((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=tf.float32)
    return x

