# tf.random.uniform((B, 768, 768, 3), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras.layers import (
    LayerNormalization, Conv2D, Softmax, ReLU,
    Layer, Input, GlobalAveragePooling2D, Dropout, Dense, ZeroPadding2D,
    DepthwiseConv2D, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


@tf.keras.utils.register_keras_serializable()
class MDTA(Layer):
    '''Multi-DConv Head Transposed Attention. 
    Channels must be divisible by num_heads.
    Includes a trainable temperature variable.'''

    def __init__(self, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        # temperature shape: (num_heads, 1, 1), initialized to 1
        self.temperature = tf.Variable(
            initial_value=tf.ones([num_heads, 1, 1], dtype=tf.float32),
            trainable=True,
            name="temperature",
            shape=[num_heads, 1, 1]
        )
        # Layers initialized in build because filters depend on input channels
        self.qkv = None
        self.qkv_conv = None
        self.project_out = None

    def build(self, input_shape):
        # input_shape: (batch_size, height, width, channels)
        _, h, w, c = input_shape

        # Define layers for qkv projection and depthwise conv
        self.qkv = Conv2D(filters=c * 3, kernel_size=1, use_bias=False)
        self.qkv_conv = Conv2D(
            filters=c * 3,
            kernel_size=3,
            padding='same',
            groups=c * 3,  # depthwise convolution as grouped conv with groups=channels
            use_bias=False,
        )
        self.project_out = Conv2D(filters=c, kernel_size=1, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (N, H, W, C)
        b, h, w, c = tf.unstack(tf.shape(inputs))
        # Compute qkv feature maps: conv1x1 then depthwise conv3x3
        qkv = self.qkv(inputs)         # (N,H,W,3C)
        qkv = self.qkv_conv(qkv)       # (N,H,W,3C)
        # Split q,k,v: each (N,H,W,C)
        q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)

        # Reshape to (N, num_heads, C_per_head, H*W)
        c_per_head = c // self.num_heads
        shape_new = [b, self.num_heads, c_per_head, h * w]
        q = tf.reshape(q, shape_new)
        k = tf.reshape(k, shape_new)
        v = tf.reshape(v, shape_new)

        # Normalize q,k on last axis
        q = tf.nn.l2_normalize(q, axis=-1)
        k = tf.nn.l2_normalize(k, axis=-1)

        # Attention: (N, num_heads, C_per_head, C_per_head)
        attn = tf.matmul(q, k, transpose_b=True)

        # Multiply with trainable temperature (broadcasted on batch dim)
        # temperature shape: (num_heads, 1, 1)
        attn = attn * self.temperature[tf.newaxis, :, :, :]

        # Softmax across last dimension
        attn = Softmax(axis=-1)(attn)

        # Multiply attention and v: (N, num_heads, C_per_head, H*W)
        out = tf.matmul(attn, v)

        # Reshape: (N, H, W, C)
        out = tf.transpose(out, perm=[0, 1, 2, 3])  # shape already correct
        out = tf.reshape(out, [b, h, w, c])

        # Final conv projection
        out = self.project_out(out)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
        })
        # We don't serialize tf.Variable attributes directly here since they are tracked automatically
        return config


@tf.keras.utils.register_keras_serializable()
class GDFN(Layer):
    '''Gated Depthwise FeedForward Network'''

    def __init__(self, expansion_factor=2, **kwargs):
        super().__init__(**kwargs)
        self.expansion_factor = expansion_factor
        self.project_in = None
        self.conv = None
        self.project_out = None

    def build(self, input_shape):
        _, h, w, c = input_shape
        hidden_channels = int(c * self.expansion_factor)
        self.project_in = Conv2D(hidden_channels * 2, kernel_size=1, use_bias=False)
        self.conv = Conv2D(
            hidden_channels * 2,
            kernel_size=3,
            padding='same',
            groups=hidden_channels * 2,
            use_bias=False,
        )
        self.project_out = Conv2D(c, kernel_size=1, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        x = self.project_in(inputs)  # (N,H,W,2*C*r)
        x = self.conv(x)             # (N,H,W,2*C*r)
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)  # two halves
        gated = ReLU()(x1) * x2     # Elementwise gating
        out = self.project_out(gated)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'expansion_factor': self.expansion_factor,
        })
        return config


def _transformer_block(inputs, num_heads, expansion_factor):
    '''Transformer block with MDTA and GDFN layers'''
    b, h, w, c = inputs.shape
    assert c % num_heads == 0

    norm1 = LayerNormalization()
    attn = MDTA(num_heads)
    norm2 = LayerNormalization()
    ffn = GDFN(expansion_factor=expansion_factor)

    # Normalize and reshape for attention
    inputs_norm1 = norm1(tf.reshape(inputs, [-1, h * w, c]))
    inputs_norm1 = tf.reshape(inputs_norm1, [-1, h, w, c])
    x = inputs + attn(inputs_norm1)

    # Normalize and reshape for FFN
    inputs_norm2 = norm2(tf.reshape(x, [-1, h * w, c]))
    inputs_norm2 = tf.reshape(inputs_norm2, [-1, h, w, c])
    out = x + ffn(inputs_norm2)

    return out


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = int(filters * alpha)
    x = Conv2D(
        filters,
        kernel,
        padding="same",
        use_bias=False,
        strides=strides,
        name="conv1",
    )(inputs)
    x = BatchNormalization(axis=channel_axis, name="conv1_bn")(x)
    return ReLU(6.0, name="conv1_relu")(x)


def _depthwise_conv_block(
    inputs,
    pointwise_conv_filters,
    alpha,
    depth_multiplier=1,
    strides=(1, 1),
    block_id=1,
):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = ZeroPadding2D(((0, 1), (0, 1)), name="conv_pad_%d" % block_id)(inputs)
    x = DepthwiseConv2D(
        (3, 3),
        padding="same" if strides == (1, 1) else "valid",
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name="conv_dw_%d" % block_id,
    )(x)

    x = BatchNormalization(axis=channel_axis, name="conv_dw_%d_bn" % block_id)(x)
    x = ReLU(6.0, name="conv_dw_%d_relu" % block_id)(x)

    x = Conv2D(
        pointwise_conv_filters,
        (1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1),
        name="conv_pw_%d" % block_id,
    )(x)
    x = BatchNormalization(axis=channel_axis, name="conv_pw_%d_bn" % block_id)(x)
    return ReLU(6.0, name="conv_pw_%d_relu" % block_id)(x)


class MyModel(tf.keras.Model):
    '''MobileNetV1 with integrated MDTA + Transformer blocks per issue code'''

    def __init__(self, input_shape=(768, 768, 3), alpha=1.0,
                 depth_multiplier=1, dropout=0.3, num_classes=6):
        super().__init__()
        self.alpha = alpha
        self.depth_multiplier = depth_multiplier
        self.dropout = dropout
        self.num_classes = num_classes
        self.input_shape_ = input_shape
        self.num_heads = 4
        self.expansion_factor = 3

        # Build layers very similarly to MobileNet with inserted Transformer blocks
        self.conv1 = lambda x: _conv_block(x, 32, alpha, strides=(2, 2))
        self.trans1 = lambda x: _transformer_block(x, self.num_heads, self.expansion_factor)
        self.dconv2 = lambda x: _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
        self.trans2 = lambda x: _transformer_block(x, self.num_heads, self.expansion_factor)
        self.dconv3 = lambda x: _depthwise_conv_block(
            x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
        self.trans3 = lambda x: _transformer_block(x, self.num_heads, self.expansion_factor)
        self.dconv4 = lambda x: _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
        self.trans4 = lambda x: _transformer_block(x, self.num_heads, self.expansion_factor)
        self.dconv5 = lambda x: _depthwise_conv_block(
            x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
        self.trans5 = lambda x: _transformer_block(x, self.num_heads, self.expansion_factor)
        self.dconv6 = lambda x: _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
        self.trans6 = lambda x: _transformer_block(x, self.num_heads, self.expansion_factor)
        self.dconv7 = lambda x: _depthwise_conv_block(
            x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
        self.dconv8 = lambda x: _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
        self.dconv9 = lambda x: _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
        self.dconv10 = lambda x: _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
        self.dconv11 = lambda x: _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
        self.dconv12 = lambda x: _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
        self.dconv13 = lambda x: _depthwise_conv_block(
            x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
        self.dconv14 = lambda x: _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

        self.global_pool = GlobalAveragePooling2D(keepdims=True)
        self.dropout_layer = Dropout(dropout)
        self.conv_preds = Conv2D(num_classes, (1, 1), padding="same")
        self.reshape_layer = tf.keras.layers.Reshape((num_classes,))
        self.activation = tf.keras.layers.Activation('softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.trans1(x)
        x = self.dconv2(x)
        x = self.trans2(x)
        x = self.dconv3(x)
        x = self.trans3(x)
        x = self.dconv4(x)
        x = self.trans4(x)
        x = self.dconv5(x)
        x = self.trans5(x)
        x = self.dconv6(x)
        x = self.trans6(x)
        x = self.dconv7(x)
        x = self.dconv8(x)
        x = self.dconv9(x)
        x = self.dconv10(x)
        x = self.dconv11(x)
        x = self.dconv12(x)
        x = self.dconv13(x)
        x = self.dconv14(x)

        x = self.global_pool(x)
        x = self.dropout_layer(x)
        x = self.conv_preds(x)    # (batch, 1, 1, num_classes)
        x = self.reshape_layer(x) # (batch, num_classes)
        x = self.activation(x)
        return x


def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel(input_shape=(768, 768, 3), dropout=0.3, num_classes=6)


def GetInput():
    # Return a random tensor input that matches MyModel input shape
    # Shape: (batch_size, height, width, channels)
    batch_size = 2  # arbitrary batch size
    input_shape = (batch_size, 768, 768, 3)
    return tf.random.uniform(input_shape, dtype=tf.float32)

