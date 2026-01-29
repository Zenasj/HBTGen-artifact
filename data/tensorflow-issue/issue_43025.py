# tf.random.uniform((B, None, None, 3), dtype=tf.float32) ← Inferred input shape based on example code with 3 channels and spatial dims unknown

import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder layers matching example convolutional stack
        self.encoder_layers = [
            tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME"),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME"),
            tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME"),
            tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="encoded_layer")
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        # Decoder layers matching example Conv2DTranspose stack
        self.decoder_layers = [
            tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="second_section"),
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME"),
            tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME"),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(2, 2), padding="SAME")
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.decoder_layers:
            x = layer(x)
        return x


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs, start_from_encoded=False):
        """
        Forward pass through the full model or decoder starting from encoded representation.
        
        Args:
            inputs: Tensor input, if start_from_encoded is False this should be the images,
                    if True this should be the encoded tensor input for the decoder.
            start_from_encoded: bool, whether to start forward pass from encoder or decoder.
        
        Returns:
            Tensor output of the full model or decoder output.
        """
        if start_from_encoded:
            # inputs are already encoded representation -> decode only
            return self.decoder(inputs)
        else:
            # Full forward pass: encode then decode
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            return decoded


def my_model_function():
    """
    Returns an instance of the MyModel class.
    """
    return MyModel()


def GetInput():
    """
    Generate a random input tensor matching the expected input shape:
    Batch size: 1 (arbitrary for testing), Height & Width: 64x64 (chosen arbitrary spatial dim),
    Channels: 3 (color channels as per example).
    
    Returns:
        A tf.Tensor with shape (1, 64, 64, 3), dtype tf.float32.
    """
    return tf.random.uniform((1, 64, 64, 3), dtype=tf.float32)

