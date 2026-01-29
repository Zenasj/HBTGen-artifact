# tf.random.uniform((batch, 448, 448, 1), dtype=tf.float32) ← input shape assumed from model input layers and data processing

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Encoder A: Conv2D layers with tanh activation and specific initializers
        self.encoder_a = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            )
        ])

        # Encoder B: Conv2D layers followed by Dense layers, all with tanh or swish activations
        self.encoder_b = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                1024, activation="swish",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.04),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Dense(
                512, activation="swish",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.04),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Dense(
                256, activation="swish",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.04),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            )
        ])

        # Decoder: Conv2DTranspose layers with tanh except last sigmoid output layer (producing 2 channels)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(
                filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=2, kernel_size=(3,3), strides=(1,1), padding="same", activation="sigmoid",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                bias_initializer=tf.keras.initializers.Constant(0.1)
            )
        ])

    def fusion(self, a, b):
        # Fusion implemented as concatenation along channel axis (last dim)
        return tf.keras.layers.Concatenate(axis=3)([a, b])

    def call(self, x):
        """
        Forward pass:
        1. Encode input x using encoder_a → feature map tensor
        2. Encode output of encoder_a using encoder_b → vector tensor
        3. Reshape encoder_b output to match spatial dims of encoder_a output to concatenate
        4. Concatenate (fuse) encoded feature map and reshaped global features
        5. Decode the fused features to get output, expected shape (batch, 448, 448, 2)
        """
        encoded_a = self.encoder_a(x)  # shape: (batch, H, W, C_a)
        encoded_b = self.encoder_b(encoded_a)  # shape: (batch, 256) from Dense layers
        
        batch_size = tf.shape(encoded_a)[0]
        spatial_dims = tf.shape(encoded_a)[1:3]  # (H, W)

        # reshape encoded_b to (batch, H, W, -1)
        # encoded_b shape is (batch, 256), expand dims and tile spatially
        # This is inferred to broadcast the global features spatially for concatenation
        encoded_b_reshaped = tf.reshape(encoded_b, [batch_size, 1, 1, -1])
        encoded_b_tiled = tf.tile(encoded_b_reshaped, [1, spatial_dims[0], spatial_dims[1], 1])

        fused = self.fusion(encoded_a, encoded_b_tiled)  # concat channels = C_a + 256

        decoded = self.decoder(fused)
        # Output shape expected (batch, 448, 448, 2): color channels ab in autoencoder

        return decoded


def my_model_function():
    # Return an instance of the Autoencoder-based model
    model = MyModel()
    # It's typical to build the model by calling once or specifying input shape
    dummy_input = tf.random.uniform((1, 448, 448, 1), dtype=tf.float32)
    model(dummy_input)  # Build model
    return model


def GetInput():
    # Return random input in correct shape: (batch=1, 448, 448, 1), float32
    # normalized approximately between 0 and 1 since input used L channel normalized by 255
    return tf.random.uniform((1, 448, 448, 1), minval=0, maxval=1, dtype=tf.float32)

