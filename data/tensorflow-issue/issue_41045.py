# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)  # CIFAR input assumed (batch, height=32, width=32, channels=3)

import tensorflow as tf

class BottleNeck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(1, 1), use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(3, 3), strides=(stride, stride), use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=out_channels * BottleNeck.expansion, kernel_size=(1, 1), use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()

        # Downsample branch as Sequential if needed
        self.downsample = tf.keras.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.downsample.add(
                tf.keras.layers.Conv2D(filters=out_channels * BottleNeck.expansion, kernel_size=(1, 1), strides=(stride, stride), use_bias=False)
            )
            self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.pad1(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        down = self.downsample(inputs, training=training)
        out += down
        out = tf.nn.relu(out)

        return out


class ResNet(tf.keras.Model):
    def __init__(self, dataset, block, num_blocks, num_classes):
        super(ResNet, self).__init__()

        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            # CIFAR dataset specific model parameters
            self.in_channels = 64
            self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
            self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), use_bias=False)
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.relu = tf.nn.relu

            # Create 4 layers (sequential blocks)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
            self.fc = tf.keras.layers.Dense(num_classes)

        else:
            # Placeholder for other datasets (not in original snippet)
            raise NotImplementedError("Dataset {} not implemented".format(dataset))

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = tf.keras.Sequential()
        for stride in strides:
            layers.add(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return layers

    def call(self, x, training=None):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.pad1(x)
            x = self.conv1(x)
            x = self.bn1(x, training=training)
            x = self.relu(x)

            x = self.layer1(x, training=training)
            x = self.layer2(x, training=training)
            x = self.layer3(x, training=training)
            x = self.layer4(x, training=training)

            x = self.avgpool(x)
            x = tf.keras.layers.Flatten()(x)
            x = self.fc(x)
            return x
        else:
            raise NotImplementedError("Forward pass not implemented for dataset {}".format(self.dataset))


# Since the original issue reports loading problems when using sequential blocks inside subclassed model,
# but no direct comparison or fusion of multiple models is described,
# we will just implement the ResNet model faithfully as above in MyModel.
# This preserves the sequential blocks, as original user code does.

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Construct a ResNet for CIFAR-10 with BottleNeck blocks and 3,4,6,3 layers (ResNet50 style)
        # We pick cifar10 dataset (3 channels, 10 classes)
        self.resnet = ResNet(dataset='cifar10', block=BottleNeck, num_blocks=[3,4,6,3], num_classes=10)

    def call(self, inputs, training=None):
        return self.resnet(inputs, training=training)


def my_model_function():
    # Return an instance of MyModel; no pretrained weights since none provided.
    return MyModel()


def GetInput():
    # Return a random tensor input matching expected CIFAR-10 input shape (batch=1 for simplest testing).
    # dtype tf.float32 consistent with layers.
    return tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)

