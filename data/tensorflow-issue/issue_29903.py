# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← typical input for the DRN model is (batch, height, width, channels) = (4, 224, 224, 3)

import tensorflow as tf
import numpy as np

def conv3x3(out_planes, data_format='channels_last', stride=1, padding='same', dilation=1, name=None, use_bias=False):
    """3x3 convolution with padding"""
    return tf.keras.layers.Conv2D(
        filters=out_planes,
        kernel_size=3,
        data_format=data_format,
        strides=(stride, stride),
        padding=padding,
        use_bias=use_bias,
        dilation_rate=(dilation, dilation),
        kernel_initializer=tf.initializers.he_normal(),
        name=name)

def conv1x1(out_planes, data_format='channels_last', padding='same', stride=1):
    """1x1 convolution"""
    return tf.keras.layers.Conv2D(
        filters=out_planes,
        kernel_size=1,
        strides=(stride, stride),
        data_format=data_format,
        padding=padding,
        use_bias=False,
        kernel_initializer=tf.initializers.he_normal())


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, planes=1, stride=1, data_format='channels_last', downsample=None,
                 dilation=(1, 1), residual=True, key=None, stage=None):
        super(BasicBlock, self).__init__()
        self.data_format = data_format
        bn_axis = 1 if self.data_format == 'channels_first' else 3
        self.conv1 = conv3x3(
            out_planes=planes, stride=stride, padding='same',
            data_format=self.data_format, dilation=dilation[0],
            name='{}_{}_conv0'.format(key, stage))

        self.bn1 = tf.keras.layers.BatchNormalization(axis=bn_axis, name='{}_{}_BN0'.format(key, stage))

        self.conv2 = conv3x3(
            out_planes=planes, padding='same',
            data_format=self.data_format, dilation=dilation[0],
            name='{}_{}_conv1'.format(key, stage))

        self.bn2 = tf.keras.layers.BatchNormalization(axis=bn_axis, name='{}_{}_BN1'.format(key, stage))

        self.downsample = downsample
        self.relu = tf.keras.layers.ReLU(name='{}_{}_Relu'.format(key, stage))
        self.stride = stride
        self.residual = residual

    def call(self, inputs, training=None):
        residual = inputs
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, planes, stride=1, data_format='channels_last', downsample=None, dilation=(1, 1)):
        super(Bottleneck, self).__init__()
        bn_axis = 1 if data_format == 'channels_first' else 3
        self.conv1 = conv1x1(planes, data_format=data_format)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=bn_axis)
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = conv3x3(planes, stride, padding='same', dilation=dilation[1], data_format=data_format)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=bn_axis)
        self.conv3 = conv1x1(planes * 4, data_format=data_format)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=bn_axis)
        self.downsample = downsample
        self.stride = stride

    def call(self, inputs, training=None):
        identity = inputs
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(inputs)
        out += identity
        out = self.relu(out)
        return out


class pooling(tf.keras.Model):
    def __init__(self, pool_size, stride=None, data_format='channels_last'):
        super(pooling, self).__init__()
        self.pool_size = pool_size
        self.data_format = data_format
        self.stride = stride if stride is not None else self.pool_size

    def call(self, inputs):
        # tf.layers.average_pooling2d was TF1.x, so we use tf.keras.layers.AveragePooling2D here
        avgpool = tf.keras.layers.AveragePooling2D(
            pool_size=self.pool_size,
            strides=self.stride,
            padding='valid',
            data_format=self.data_format)
        return avgpool(inputs)


class DRN(tf.keras.Model):
    def __init__(self, block, layers, data_format='channels_last', num_classes=7,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch
        self.poolsize = pool_size
        self.data_format = data_format
        self.bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv0 = tf.keras.layers.Conv2D(
            filters=channels[0], kernel_size=7, strides=1, padding='same',
            use_bias=False, data_format=self.data_format,
            kernel_initializer=tf.initializers.he_normal(), name='L0_conv0')
        self.bn0 = tf.keras.layers.BatchNormalization(axis=self.bn_axis, name='L0_BN0')
        self.relu0 = tf.keras.layers.ReLU(name='L0_Relu0')

        if arch == 'C':
            self.layer1 = self._make_layer(
                block=BasicBlock, planes=channels[0], blocks=layers[0], stride=1,
                data_format=self.data_format, key='CL1')
            self.layer2 = self._make_layer(
                block=BasicBlock, planes=channels[1], blocks=layers[1], stride=2,
                data_format=self.data_format, key='CL2')
        elif arch == 'D':
            self.layer1 = self._make_conv_layers(
                channels=channels[0], convs=layers[0], stride=1,
                data_format=self.data_format, key='DL1')
            self.layer2 = self._make_conv_layers(
                channels=channels[1], convs=layers[1], stride=2,
                data_format=self.data_format, key='DL2')

        self.layer3 = self._make_layer(
            block=block, planes=channels[2], blocks=layers[2], stride=2,
            data_format=self.data_format, key='L3')
        self.layer4 = self._make_layer(
            block=block, planes=channels[3], blocks=layers[3], stride=2,
            data_format=self.data_format, key='L4')
        self.layer5 = self._make_layer(
            block=block, planes=channels[4], blocks=layers[4], dilation=2,
            new_level=False, data_format=self.data_format, key='L5')
        self.layer6 = None if layers[5] == 0 else self._make_layer(
            block, channels[5], layers[5], dilation=4,
            new_level=False, data_format=self.data_format, key='L6')

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else self._make_layer(
                BasicBlock, channels[6], layers[6], dilation=2,
                new_level=False, residual=False,
                data_format=self.data_format, key='CL7')
            self.layer8 = None if layers[7] == 0 else self._make_layer(
                BasicBlock, channels[7], layers[7], dilation=1,
                new_level=False, residual=False,
                data_format=self.data_format, key='CL8')
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else self._make_conv_layers(
                channels[6], layers[6], dilation=2,
                data_format=self.data_format, key='DL7')
            self.layer8 = None if layers[7] == 0 else self._make_conv_layers(
                channels[7], layers[7], dilation=1,
                data_format=self.data_format, key='DL8')

        if num_classes > 0:
            self.avgpool = tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format)
            self.fc = tf.keras.layers.Dense(units=num_classes)

    def _make_layer(self, block, planes, blocks, stride=1,
                    dilation=1, new_level=True,
                    data_format='channels_last', residual=True, key=None):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                conv1x1(out_planes=planes * block.expansion, stride=stride, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=self.bn_axis)
            ], name='downsample')

        layers = []
        layers.append(block(
            planes=planes, stride=stride, downsample=downsample,
            dilation=(1, 1) if dilation == 1 else
            (dilation // 2 if new_level else dilation, dilation),
            data_format=data_format, residual=residual, key=key, stage='0'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes, residual=residual,
                                dilation=(dilation, dilation),
                                data_format=data_format, key=key, stage=i))
        return tf.keras.Sequential(layers, name=key)

    def _make_conv_layers(self, channels, convs, stride=1,
                          dilation=1, data_format='channels_last', key=None):
        modules = []
        for i in range(convs):
            modules.extend([
                conv3x3(out_planes=channels, stride=stride if i == 0 else 1,
                        padding='same', use_bias=False, dilation=dilation,
                        data_format=data_format, name='{}_{}_Conv'.format(key, i)),
                tf.keras.layers.BatchNormalization(axis=self.bn_axis, name='{}_{}_BN'.format(key, i)),
                tf.keras.layers.ReLU(name='{}_{}_Relu'.format(key, i))])
            self.inplanes = channels
        return tf.keras.Sequential(modules, name=key)

    def call(self, x, training=None):
        x = self.conv0(x)
        x = self.bn0(x, training=training)
        x = self.relu0(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.layer5(x, training=training)

        if self.layer6 is not None:
            x = self.layer6(x, training=training)

        if self.layer7 is not None:
            x = self.layer7(x)
        if self.layer8 is not None:
            x = self.layer8(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
        return x


class MyModel(tf.keras.Model):
    """
    Wrapper model that encapsulates the DRN with BasicBlock,
    along with a learning rate scheduler function analogous to tf.train.piecewise_constant
    for a multi-step decay schedule.

    This fusion represents both the DRN and the learning rate scheduling logic
    mentioned in the original issue discussion.
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # Instantiate DRN model with BasicBlock and a default layer configuration:
        # Using smaller layers to keep it reasonable for demo purposes.
        self.drn = DRN(
            block=BasicBlock,
            layers=[1, 1, 2, 2, 2, 2, 1, 1],
            arch='C',
            num_classes=4  # Common example output classes
        )
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        # Scheduler parameters:
        self.n_step_epoch = 10
        self.init_lr = 0.1
        self.policy = 'multistep_2_5'  # corresponds to decay at milestone epochs 2 and 5

        # Precompute boundaries and values as tensors for piecewise_constant call:
        milestones = [2,5]  # extracted from policy string 'multistep_2_5'
        self.boundaries = [self.n_step_epoch * m for m in milestones]  # [20, 50]
        # Note: The original code had an error computing values /= 0.1**-i. Here we fix it:
        # values = [init_lr] + [init_lr * (0.1**i) for i in range(1, len(milestones)+1)]
        self.values = [self.init_lr] + [self.init_lr * (0.1 ** i) for i in range(1, len(milestones) + 1)]

    def learning_rate(self):
        # Return the learning rate for current global_step using tf.train.piecewise_constant
        # tf.train.piecewise_constant expects boundaries and values to be lists
        return tf.train.piecewise_constant(tf.cast(self.global_step, tf.int32), self.boundaries, self.values)

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # Forward pass returns class logits from DRN
        return self.drn(inputs, training=training)

    def increment_step(self):
        self.global_step.assign_add(1)


def my_model_function():
    # Return an instance of MyModel, ready to use
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the expected input of MyModel (DRN)
    # Batch size: 4, Image size: 224x224, Channels: 3, dtype: tf.float32
    return tf.random.uniform((4, 224, 224, 3), dtype=tf.float32)

