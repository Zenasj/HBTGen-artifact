# tf.random.uniform((1, 512, 512, 3), dtype=tf.float32) ← Input shape inferred from example usage in the issue 

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    """
    UNet-like architecture implemented as a Keras Model subclass.
    This model uses encoder and decoder blocks with Conv2D and Conv2DTranspose layers
    along with batch normalization, dropout, and nonlinearities.

    The original issue was that trainable variables were empty when layers were appended 
    to lists initialized as instance attributes first, instead of building lists locally 
    and assigning them as instance attributes at the end of __init__.
    This implementation fixes that by setting encoder_layers, decoder_layers,
    and concat_layers attributes after construction, ensuring Keras tracks the sublayers properly.
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

        # Store params
        self.use_dropout_encoder = use_dropout_encoder
        self.use_dropout_decoder = use_dropout_decoder
        self.dropout_probability = dropout_prob
        self.encoder_non_linearity = encoder_non_linearity
        self.decoder_non_linearity = decoder_non_linearity
        self.kernel_size = kernel_size

        # Important: build lists locally first, assign to self at the end
        encoder_layers = []
        decoder_layers = []
        concat_layers = []

        # Encoder layers spec - identical to original
        encoder_layers_spec = [128, 256, 512, 512, 512, 512, 512, 512]

        for i, filters in enumerate(encoder_layers_spec):
            block = self.get_encoder_block(filters, use_bn=(i != 0))
            encoder_layers.append(block)

        # Decoder layers spec
        decoder_layer_spec = [512, 512, 512, 512, 512, 256, 128]

        for i, filters in enumerate(decoder_layer_spec):
            concat_layers.append(keras.layers.Concatenate())
            block = self.get_decoder_block(filters, use_dropout=(i < 3))
            decoder_layers.append(block)

        initializer = tf.random_normal_initializer(0.0, 0.02)
        self.final_layer = keras.layers.Conv2DTranspose(
            channels,
            self.kernel_size,
            strides=(2, 2),
            padding="same",
            activation=keras.activations.tanh,
            kernel_initializer=initializer,
        )

        # Assign to instance attributes *after* list creation (fix the trackability issue)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.concat_layers = concat_layers

    def get_block(
        self,
        filters,
        conv_layer=None,
        use_bn=True,
        use_dropout=False,
        non_linearity=keras.layers.LeakyReLU,
    ):
        initializer = tf.random_normal_initializer(0.0, 0.02)
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

        if use_bn:
            block.append(keras.layers.BatchNormalization())

        if use_dropout:
            block.append(keras.layers.Dropout(self.dropout_probability))

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

    def call(self, inputs, training=True):
        # Evaluate encoder blocks sequentially, save intermediate outputs
        x = inputs
        encoder_layer_eval = []

        for block in self.encoder_layers:
            for layer in block:
                # BatchNorm and Dropout layers receive training flag
                if isinstance(layer, keras.layers.BatchNormalization) or isinstance(layer, keras.layers.Dropout):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
            encoder_layer_eval.append(x)

        # Remove last encoder layer output (matches original code)
        encoder_layer_eval = encoder_layer_eval[:-1]

        # Decode by going through decoder blocks and concatenate with encoder layers
        for i, block in enumerate(self.decoder_layers):
            for layer in block:
                if isinstance(layer, keras.layers.BatchNormalization) or isinstance(layer, keras.layers.Dropout):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
            x = self.concat_layers[i]([x, encoder_layer_eval[-1 - i]])

        x = self.final_layer(x)
        return x


def my_model_function():
    # Return an instance of MyModel, with example parameters consistent with original
    return MyModel(
        input_res=512,
        min_res=4,
        kernel_size=4,
        initial_filters=64,
        filters_cap=512,
        channels=3,
    )


def GetInput():
    # Return a random tensor compatible with the model's expected input
    # Shape: (batch=1, height=512, width=512, channels=3), dtype float32 per example in the issue
    return tf.random.uniform((1, 512, 512, 3), dtype=tf.float32)

