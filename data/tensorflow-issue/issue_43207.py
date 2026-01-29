# tf.random.uniform((1, 288, 224, 3), dtype=tf.float32) ← Assumed input shape and dtype from the model definition

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, nClasses=25, input_height=288, input_width=224, n_filters=64, dropout=0.2,
                 batchnorm=True, activation=True):
        super().__init__()
        self.nClasses = nClasses
        self.input_height = input_height
        self.input_width = input_width
        self.n_filters = n_filters
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.activation = activation

        # Define the conv2d_block_3layers as layers since subclassing
        def conv2d_block_3layers():
            block = tf.keras.Sequential()
            for _ in range(3):
                block.add(layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same'))
                if batchnorm:
                    block.add(layers.BatchNormalization())
                if activation:
                    block.add(layers.Activation('relu'))
                # Dropout after first two conv layers but not last could be debated.
                # Following original code, Dropout after first and second conv layer, no dropout after last.
                # However, original fn adds dropout after first and second Conv2Ds but not after the third, here simplified.
                block.add(layers.Dropout(dropout))
            return block

        # We'll define conv2d_block_3layers as a method for modularity 
        # (but here keeping as Sequential for simplicity)
        # But due to the dropout placement differences, manually unrolling to match original per layer pattern is best.

        # Build layers

        # Because conv2d_block_3layers applies dropout only on the first and second conv, 
        # we'll rebuild conv2d_block_3layers with exact layers.

        def conv_block(n_filters):
            layers_list = []
            # Conv 1
            layers_list.append(layers.Conv2D(filters=n_filters, kernel_size=(3,3), padding='same'))
            if batchnorm:
                layers_list.append(layers.BatchNormalization())
            if activation:
                layers_list.append(layers.Activation('relu'))
            layers_list.append(layers.Dropout(dropout))
            # Conv 2
            layers_list.append(layers.Conv2D(filters=n_filters, kernel_size=(3,3), padding='same'))
            if batchnorm:
                layers_list.append(layers.BatchNormalization())
            if activation:
                layers_list.append(layers.Activation('relu'))
            layers_list.append(layers.Dropout(dropout))
            # Conv 3
            layers_list.append(layers.Conv2D(filters=n_filters, kernel_size=(3,3), padding='same'))
            if batchnorm:
                layers_list.append(layers.BatchNormalization())
            if activation:
                layers_list.append(layers.Activation('relu'))
            return tf.keras.Sequential(layers_list)

        self.conv_block1 = conv_block(n_filters * 1)
        self.conv_block2 = conv_block(n_filters * 2)
        self.conv_block3 = conv_block(n_filters * 4)
        self.conv_block4 = conv_block(n_filters * 4)
        self.conv_block5 = conv_block(n_filters * 8)
        self.dropout_p5 = layers.Dropout(dropout)

        self.pool = layers.MaxPooling2D((2, 2))

        # Upsampling blocks
        self.up6 = layers.Conv2DTranspose(n_filters * 4, kernel_size=(3,3), strides=(2,2), padding="same")
        self.conv_block6 = conv_block(n_filters * 4)

        self.up7 = layers.Conv2DTranspose(n_filters * 4, kernel_size=(3,3), strides=(2,2), padding="same")
        self.conv_block7 = conv_block(n_filters * 4)

        self.up8 = layers.Conv2DTranspose(n_filters * 2, kernel_size=(3,3), strides=(2,2), padding="same")
        self.conv_block8 = conv_block(n_filters * 2)

        self.up9 = layers.Conv2DTranspose(n_filters * 1, kernel_size=(3,3), strides=(2,2), padding="same")
        self.conv_block9 = conv_block(n_filters * 1)

        self.concat = layers.Concatenate(axis=3)
        self.output_conv = layers.Conv2D(filters=nClasses, kernel_size=1, activation="softmax")

    def call(self, inputs, training=False):
        # Encoder path
        c1 = self.conv_block1(inputs, training=training)
        p1 = self.pool(c1)

        c2 = self.conv_block2(p1, training=training)
        p2 = self.pool(c2)

        c3 = self.conv_block3(p2, training=training)
        p3 = self.pool(c3)

        c4 = self.conv_block4(p3, training=training)
        p4 = self.pool(c4)

        c5 = self.conv_block5(p4, training=training)
        p5 = self.dropout_p5(c5, training=training)

        # Decoder path
        up6 = self.up6(p5)
        m6 = self.concat([up6, c4])
        c6 = self.conv_block6(m6, training=training)

        up7 = self.up7(c6)
        m7 = self.concat([up7, c3])
        c7 = self.conv_block7(m7, training=training)

        up8 = self.up8(c7)
        m8 = self.concat([up8, c2])
        c8 = self.conv_block8(m8, training=training)

        up9 = self.up9(c8)
        m9 = self.concat([up9, c1])
        c9 = self.conv_block9(m9, training=training)

        output = self.output_conv(c9)
        return output


def my_model_function():
    # Instantiate the model with default parameters matching the original example
    return MyModel()


def GetInput():
    # Generate a random tensor input matching expected input size and channels:
    # input shape: (batch=1, height=288, width=224, channels=3)
    # dtype: float32, values in range [0,1] similar to normalized images in the original representative dataset generator.
    input_tensor = tf.random.uniform((1, 288, 224, 3), dtype=tf.float32)
    return input_tensor

