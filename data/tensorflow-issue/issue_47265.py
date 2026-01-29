# tf.random.uniform((B, 416, 416, 3), dtype=tf.float32) ← inferred input shape for YOLOv3

import tensorflow as tf
import numpy as np

NUM_CLASS = 1
ANCHORS = np.array(
    [1.25, 1.625, 2.0, 3.75, 4.125, 2.875,
     1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375,
     3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875]
).reshape(3, 3, 2).astype(np.float32)
STRIDES = np.array([8, 16, 32]).astype(np.float32)
IOU_LOSS_THRESH = 0.5


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Custom BatchNorm layer that respects trainable flag in inference mode.
    """
    def call(self, inputs, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(inputs, training)


def convolutional(input_layer, filters_shape, downsample=False,
                  activate=True, bn=True):
    """
    Custom convolutional layer with batch norm and leaky relu activation.
    """
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1

    conv = tf.keras.layers.Conv2D(
        filters=filters_shape[-1],
        kernel_size=filters_shape[0],
        strides=strides,
        padding=padding,
        use_bias=not bn,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.)
    )(input_layer)

    if bn:
        conv = BatchNormalization()(conv)
    if activate:
        conv = tf.nn.leaky_relu(conv, alpha=0.1)
    return conv


def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    """
    Residual block for darknet53.
    """
    short_cut = input_layer
    conv = convolutional(input_layer,
                         filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))
    output = short_cut + conv
    return output


def darknet53(input_data):
    """
    Darknet53 backbone model.
    
    Returns route_1, route_2 feature maps and last conv output.
    """
    input_data = convolutional(input_data, (3, 3, 3, 32))
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True)

    # 1 residual block with 64 filters
    for _ in range(1):
        input_data = residual_block(input_data, 64, 32, 64)

    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True)

    # 2 residual blocks with 128 filters
    for _ in range(2):
        input_data = residual_block(input_data, 128, 64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    # 8 residual blocks with 256 filters
    for _ in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data

    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    # 8 residual blocks with 512 filters
    for _ in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data

    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    # 4 residual blocks with 1024 filters
    for _ in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data


def upsample(input_layer):
    """
    Upsample by factor of 2 using nearest neighbor.
    """
    return tf.image.resize(input_layer,
                           (input_layer.shape[1] * 2, input_layer.shape[2] * 2),
                           method='nearest')


def YOLOv3(input_layer):
    """
    YOLOv3 model returning list of 3 bbox output tensors.
    """
    route_1, route_2, conv = darknet53(input_layer)

    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))

    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(
        conv_lobj_branch,
        (1, 1, 1024, 3 * (NUM_CLASS + 5)),
        activate=False,
        bn=False
    )

    conv = convolutional(conv, (1, 1, 512, 256))
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(
        conv_mobj_branch,
        (1, 1, 512, 3 * (NUM_CLASS + 5)),
        activate=False,
        bn=False
    )

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(
        conv_sobj_branch,
        (1, 1, 256, 3 * (NUM_CLASS + 5)),
        activate=False,
        bn=False
    )

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(conv_output, i=0):
    """
    Decode conv layer output tensor to bounding box predictions.
    """
    batch_size = tf.shape(conv_output)[0]
    output_size = tf.shape(conv_output)[1]

    conv_output = tf.reshape(conv_output,
                             (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis],
                [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :],
                [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :],
                      [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def create_yolo3_model(input_size=416):
    """
    Creates the full YOLOv3 Keras model including decoding layers.
    """
    input_layer = tf.keras.layers.Input(shape=[input_size, input_size, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(inputs=input_layer, outputs=bbox_tensors)
    return model


class MyModel(tf.keras.Model):
    """
    Encapsulates the YOLOv3 model for TF 2.4+ compatibility.
    """

    def __init__(self, input_size=416):
        super().__init__()
        self.input_size = input_size
        self.yolo = create_yolo3_model(input_size)

    def call(self, inputs, training=False):
        # Inputs shape: (batch_size, 416, 416, 3)
        # Outputs: list of 3 decoded bbox tensors
        outputs = self.yolo(inputs, training=training)
        return outputs


def my_model_function():
    """
    Returns an instance of MyModel with default input size 416.
    """
    return MyModel()


def GetInput():
    """
    Returns a random tensor input suitable for MyModel.
    Batch size is 1, image size is 416x416, 3 channels, float32 in [0,1].
    """
    return tf.random.uniform((1, 416, 416, 3), dtype=tf.float32)

