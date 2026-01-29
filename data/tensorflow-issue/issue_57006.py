# tf.random.uniform((1, 320, 1024, 3), dtype=tf.float32) ← inferred input shape from original images resized to (320,1024) with 3 channels RGB

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the issue context, the model is a ResNet18 encoder + U-Net style decoder
        # We do not have the exact ResNet18 weights or decoder implementation here,
        # so we reconstruct a plausible simple ResNet18 encoder backbone + decoder structure in Keras.

        # For demonstration:
        #  - Use tf.keras.applications.ResNet50 (closest std available) as encoder substitute
        #  - Implement a simplified decoder with Conv2DTranspose and skip connections
        #  - Since the original model is from PyTorch, and input shape is (1, 3 channels last),
        #    we assume input shape (320, 1024, 3), channels-last.

        # Encoder from tf.keras.applications
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_shape=(320, 1024, 3)
        )
        # Extract layers for skip connections by layer name
        # In ResNet50:
        # conv1_relu: initial conv block output (after activation)
        # conv2_block3_out, conv3_block4_out, conv4_block6_out, conv5_block3_out: feature maps at different depths
        self.encoder = tf.keras.Model(
            inputs=base_model.input,
            outputs=[
                base_model.get_layer("conv1_relu").output,
                base_model.get_layer("conv2_block3_out").output,
                base_model.get_layer("conv3_block4_out").output,
                base_model.get_layer("conv4_block6_out").output,
                base_model.get_layer("conv5_block3_out").output,
            ],
        )

        # Decoder blocks reversely upsample + concatenate skip features
        self.up4 = tf.keras.layers.Conv2DTranspose(512, 3, strides=2, padding="same", activation="relu")
        self.skip_conv4 = tf.keras.layers.Conv2D(512, 1, padding="same", activation="relu")

        self.up3 = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")
        self.skip_conv3 = tf.keras.layers.Conv2D(256, 1, padding="same", activation="relu")

        self.up2 = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")
        self.skip_conv2 = tf.keras.layers.Conv2D(128, 1, padding="same", activation="relu")

        self.up1 = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")
        self.skip_conv1 = tf.keras.layers.Conv2D(64, 1, padding="same", activation="relu")

        # Final conv to output same channels as input (3), for image-to-image network
        self.final_conv = tf.keras.layers.Conv2D(3, 1, padding="same", activation="sigmoid")

    def call(self, inputs, training=False):
        # inputs: expected shape (batch, 320, 1024, 3)

        # Encoder forward pass with skip connections outputs
        f1, f2, f3, f4, f5 = self.encoder(inputs, training=training)
        # f5: deepest features

        # Decoder path (upsample + concat skip)

        # Decode stage 4
        x = self.up4(f5)  # Upsample f5
        s4 = self.skip_conv4(f4)
        x = tf.concat([x, s4], axis=-1)
        x = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x)

        # Decode stage 3
        x = self.up3(x)
        s3 = self.skip_conv3(f3)
        x = tf.concat([x, s3], axis=-1)
        x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)

        # Decode stage 2
        x = self.up2(x)
        s2 = self.skip_conv2(f2)
        x = tf.concat([x, s2], axis=-1)
        x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)

        # Decode stage 1
        x = self.up1(x)
        s1 = self.skip_conv1(f1)
        x = tf.concat([x, s1], axis=-1)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)

        # Final output conv layer
        output = self.final_conv(x)  # output shape (batch, 320, 1024, 3), normalized to [0,1]

        return output


def my_model_function():
    # Simply return an instance of the reconstructed MyModel.
    # Note: weights uninitialized (None), this matches the fact the original weights are not provided.
    return MyModel()


def GetInput():
    # In the original issue, input images are loaded and resized to (320, 1024) with 3 channels, then normalized.
    # The data generator transposes input from (batch, height, width, channels) to (batch, channels, width, height),
    # but typical TF models expect channels-last (batch, height, width, channels).
    #
    # Here we provide a random input tensor matching (1, 320, 1024, 3), float32, scaled [0,1].

    input_shape = (1, 320, 1024, 3)
    input_tensor = tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)
    return input_tensor

