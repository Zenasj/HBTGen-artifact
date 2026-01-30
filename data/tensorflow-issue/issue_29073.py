import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

class UNet(keras.Model):
    """
    UNet Architecture concatenating encoder and decoder

    Examples:
        * Direct Usage:

            .. testcode::

                x = tf.ones((1, 512, 512, 3))
                u_net = UNet(input_res = 512,
                             min_res=4,
                             kernel_size=4,
                             initial_filters=64,
                             filters_cap=512,
                             channels=3)
                y = u_net(x)
                print(y.shape)

            .. testoutput::
                (1, 512, 512, 3)

    """

    def __init__(
        self,
        input_res,
        min_res,
        kernel_size,
        initial_filters,
        filters_cap,
        channels,
        use_dropout_encoder=True,
        use_dropout_decoder=True,
        dropout_prob=0.3,
        encoder_non_linearity=keras.layers.LeakyReLU,
        decoder_non_linearity=keras.layers.ReLU,
    ):
        super().__init__()

        # layer specification
        self.use_dropout_encoder = use_dropout_encoder
        self.use_dropout_decoder = use_dropout_decoder
        self.dropout_probability = dropout_prob
        self.encoder_non_linearity = encoder_non_linearity
        self.decoder_non_linearity = decoder_non_linearity
        self.kernel_size = kernel_size

        # encoder layers is a list of list, each list is a "block",
        # this makes easy the creation of decoder
        self.encoder_layers = []
        self.decoder_layers = []
        self.concat_layers = []

        # ########### Encoder creation
        encoder_layers_spec = [128, 256, 512, 512, 512, 512, 512, 512]

        decoder_layer_spec = []
        for i, filters in enumerate(encoder_layers_spec):
            self.encoder_layers.append(self.get_encoder_block(filters, use_bn=(i != 0)))

        # ############## Decoder creation
        decoder_layer_spec =[512, 512, 512, 512, 512, 256, 128]

        for i, filters in enumerate(decoder_layer_spec):
            self.concat_layers.append(keras.layers.Concatenate())
            self.decoder_layers.append(
                self.get_decoder_block(filters, use_dropout=(i < 3))
            )

        # final layer
        initializer = tf.random_normal_initializer(0.0, 0.02)
        self.final_layer = keras.layers.Conv2DTranspose(
            channels,
            self.kernel_size,
            strides=(2, 2),
            padding="same",
            activation=keras.activations.tanh,
            kernel_initializer=initializer,
        )

    def get_block(
        self,
        filters,
        conv_layer=None,
        use_bn=True,
        use_dropout=False,
        non_linearity=keras.layers.LeakyReLU,
    ):
        initializer = tf.random_normal_initializer(0.0, 0.02)
        # Conv2D
        block = [
            conv_layer(
                filters,
                self.kernel_size,
                strides=(2, 2),
                padding="same",
                use_bias=False,
                kernel_initializer=initializer,
            )
        ]

        # Batch normalization
        if use_bn:
            block.append(keras.layers.BatchNormalization())

        # dropout
        if use_dropout:
            block.append(keras.layers.Dropout(self.dropout_probability))

        # Non linearity
        block.append(non_linearity())

        return block

    def get_encoder_block(self, filters, use_bn=True):
        return self.get_block(
            filters,
            conv_layer=keras.layers.Conv2D,
            use_bn=use_bn,
            use_dropout=self.use_dropout_encoder,
            non_linearity=self.encoder_non_linearity,
        )

    def get_decoder_block(self, filters, use_bn=True, use_dropout=False):
        return self.get_block(
            filters,
            conv_layer=keras.layers.Conv2DTranspose,
            use_bn=use_bn,
            use_dropout=self.use_dropout_decoder and use_dropout,
            non_linearity=self.decoder_non_linearity,
        )

    def __call__(self, inputs, training=True):
        # encoders evaluated
        encoder_layer_eval = []
        x = inputs

        for block in self.encoder_layers:
            for layer in block:
                if isinstance(layer, keras.layers.BatchNormalization) or isinstance(
                    layer, keras.layers.Dropout
                ):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
            encoder_layer_eval.append(x)

        encoder_layer_eval = encoder_layer_eval[:-1]

        for i, block in enumerate(self.decoder_layers):
            for layer in block:
                if isinstance(layer, keras.layers.BatchNormalization) or isinstance(
                    layer, keras.layers.Dropout
                ):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
            x = self.concat_layers[i]([x, encoder_layer_eval[-1 - i]])

        x = self.final_layer(x)

        return x

self.encoder_layers = []
self.decoder_layers = []
self.concat_layers = []

def __init__(
        self,
        input_res,
        min_res,
        kernel_size,
        initial_filters,
        filters_cap,
        channels,
        use_dropout_encoder=True,
        use_dropout_decoder=True,
        dropout_prob=0.3,
        encoder_non_linearity=keras.layers.LeakyReLU,
        decoder_non_linearity=keras.layers.ReLU,
    ):
        super().__init__()

        # layer specification
        self.use_dropout_encoder = use_dropout_encoder
        self.use_dropout_decoder = use_dropout_decoder
        self.dropout_probability = dropout_prob
        self.encoder_non_linearity = encoder_non_linearity
        self.decoder_non_linearity = decoder_non_linearity
        self.kernel_size = kernel_size

        # encoder layers is a list of list, each list is a "block",
        # this makes easy the creation of decoder
        # IMPORTANT! Do not initialize here instance attributes
        encoder_layers = []
        decoder_layers = []
        concat_layers = []

        # ########### Encoder creation
        encoder_layers_spec =  [128, 256, 512, 512, 512, 512, 512, 512]

        for i, filters in enumerate(encoder_layers_spec):
            block = self.get_encoder_block(filters, use_bn=(i != 0))
            encoder_layers.append(block)

        # ############## Decoder creation
        decoder_layer_spec = [512, 512, 512, 512, 512, 256, 128]

        for i, filters in enumerate(decoder_layer_spec):
            concat_layers.append(keras.layers.Concatenate())
            block = self.get_decoder_block(filters, use_dropout=(i < 3))
            decoder_layers.append(block)

        # final layer
        initializer = tf.random_normal_initializer(0.0, 0.02)
        self.final_layer = keras.layers.Conv2DTranspose(
            channels,
            self.kernel_size,
            strides=(2, 2),
            padding="same",
            activation=keras.activations.tanh,
            kernel_initializer=initializer,
        )
       # intialize here object attributes!!!!!!!
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.concat_layers = concat_layers

class M(tf.keras.models.Model):

    def __init__(self):
        super(M, self).__init__()
        self._list = []
        self._list.append([tf.keras.layers.Dense(5), tf.keras.layers.Dense(5)])
        self._list.append([tf.keras.layers.Dense(5), tf.keras.layers.Dense(5)])

    def call(self, inputs):
        output = inputs
        for l_list in self._list:
            for l in l_list:
                output = l(output)
        return output

m = M()
m(tf.ones((10, 10)))
print(len(m.trainable_variables)) # Got 8