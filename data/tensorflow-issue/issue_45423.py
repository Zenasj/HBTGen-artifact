# tf.random.uniform((B, H, W, C), dtype=tf.int32) with B=1000, H=128, W=128, C=3 inferred from foo_volume tensor shape (1000, 2, 128, 128, 3) and input to model of shape (128,128,3)

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original sequential model structure that caused cudnn/cublas allocation issues
        # Input shape used later is (128,128,3)
        cnn_filters = [64, 64, 64]

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(128, 128, 3))

        # Conv2D -> BatchNorm layers repeated per cnn_filters
        self.conv_bn_blocks1 = []
        for filters in cnn_filters:
            self.conv_bn_blocks1.append(tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
            self.conv_bn_blocks1.append(tf.keras.layers.BatchNormalization())

        self.up_sampling = tf.keras.layers.UpSampling2D((2, 2))
        self.bn_after_upsampling = tf.keras.layers.BatchNormalization()

        # Conv2D -> BatchNorm layers in reverse filters (cnn_filters[::-1]) repeated twice based on code chunks
        self.conv_bn_blocks2 = []
        for filters in reversed(cnn_filters):
            self.conv_bn_blocks2.append(tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
            self.conv_bn_blocks2.append(tf.keras.layers.BatchNormalization())
        # Following MaxPooling and BatchNorm
        self.max_pooling = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.bn_after_pooling = tf.keras.layers.BatchNormalization()

        # An additional Conv2D -> BatchNorm per cnn_filters before final Dense (producing output shape with 3 channels)
        self.conv_bn_blocks3 = []
        for filters in cnn_filters:
            self.conv_bn_blocks3.append(tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
            self.conv_bn_blocks3.append(tf.keras.layers.BatchNormalization())

        # The final Dense layer mapped to 3 channels as per summary (Dense(3) on last conv output)
        # Since Dense on 4D tensors isn't standard, we infer it is a TimeDistributed/Dense applied per pixel, 
        # but to keep simplicity and match the original code, we implement it as Conv2D with kernel 1x1 and 3 output filters (common approximation)
        # The original has Dense(3) after conv layers, this can be approximated by Conv2D(3, 1x1)
        self.final_conv = tf.keras.layers.Conv2D(3, (1, 1), activation='linear')

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
        # First conv batchnorm blocks
        for layer in self.conv_bn_blocks1:
            x = layer(x, training=training)
        # UpSampling + BatchNorm
        x = self.up_sampling(x)
        x = self.bn_after_upsampling(x, training=training)
        # Second conv batchnorm blocks
        for layer in self.conv_bn_blocks2:
            x = layer(x, training=training)
        # MaxPool and BatchNorm
        x = self.max_pooling(x)
        x = self.bn_after_pooling(x, training=training)
        # Third conv batchnorm blocks
        for layer in self.conv_bn_blocks3:
            x = layer(x, training=training)
        # Final conv to 3 channels
        x = self.final_conv(x)
        return x


def my_model_function():
    # Instantiate the model
    model = MyModel()
    # Build the model by calling once with correct input shape
    dummy_input = tf.zeros((1, 128, 128, 3), dtype=tf.float32)
    model(dummy_input)

    # Compile the model similar to original (SGD momentum=0.05, mse loss)
    optmz = tf.keras.optimizers.SGD(momentum=0.05)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optmz, loss=loss)

    return model


def GetInput():
    # Return a random tensor shaped (batch, height, width, channels) with values roughly matching original range 0-255 and dtype int32
    # Batch size arbitrary pick 4 for demonstration, compatible with model input (128,128,3)
    batch_size = 4
    input_tensor = tf.random.uniform(
        (batch_size, 128, 128, 3), minval=0, maxval=255, dtype=tf.int32
    )
    # Normalize input to float32 0~1, since typical Keras conv expects float input (original code also normalizes to [0,1])
    input_tensor = tf.cast(input_tensor, tf.float32) / 255.0
    return input_tensor

