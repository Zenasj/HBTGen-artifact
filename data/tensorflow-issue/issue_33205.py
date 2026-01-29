# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assumed input shape: batch size B, 224x224 RGB images, channels_last format

import tensorflow as tf

cfg_imagenet = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

class MyModel(tf.keras.Model):
    def __init__(self, data_format='channels_last'):
        super(MyModel, self).__init__()
        # Use channels_last for typical TensorFlow image input: (B, H, W, C)
        self.data_format = data_format
        self.channel_axis = -1 if data_format == 'channels_last' else 1
        self.initializer = tf.keras.initializers.HeNormal()

        self.features = self._make_layers()
        # 10 classes output layer
        self.classifier = tf.keras.layers.Dense(10, kernel_initializer=self.initializer)

    def _make_layers(self):
        layers = []
        # Initial conv block
        layers.append(tf.keras.layers.Conv2D(
            32, kernel_size=3, strides=2, padding='same', kernel_initializer=self.initializer, name='conv1'))
        layers.append(tf.keras.layers.BatchNormalization(axis=self.channel_axis, name='conv1_bn'))
        layers.append(tf.keras.layers.ReLU(name='conv1_relu'))

        for idx, x in enumerate(cfg_imagenet):
            i = idx + 1
            if isinstance(x, int):
                filters = x
                strides = 1
            else:
                filters, strides = x

            layers.append(tf.keras.layers.DepthwiseConv2D(
                (3, 3), strides=strides, padding='same',
                depthwise_initializer=self.initializer,
                name=f'block{i}_dw'))
            layers.append(tf.keras.layers.BatchNormalization(axis=self.channel_axis, name=f'block{i}_bn1'))
            layers.append(tf.keras.layers.ReLU(name=f'block{i}_relu1'))
            layers.append(tf.keras.layers.Conv2D(
                filters, 1, padding='same',
                kernel_initializer=self.initializer,
                name=f'block{i}_pw'))
            layers.append(tf.keras.layers.BatchNormalization(axis=self.channel_axis, name=f'block{i}_bn2'))
            layers.append(tf.keras.layers.ReLU(name=f'block{i}_relu2'))

        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        return layers

    def call(self, input_tensor, training=False):
        x = input_tensor
        # Pass 'training' argument to BatchNormalization layers to activate update of statistics during training
        for layer in self.features:
            # BatchNorm layers require training flag to behave correctly
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        x = self.classifier(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor with batch size 8, height & width 224, and 3 color channels 
    # (Typical ImageNet style input shape, inferred from cfg_imagenet downsampling)
    # dtype float32 for images is standard
    return tf.random.uniform(shape=(8, 224, 224, 3), dtype=tf.float32)

