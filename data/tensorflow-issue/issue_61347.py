from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# import tensorflow.compat.v2 as tf
import tensorflow as tf
import keras
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

from keras.layers import Layer, Activation, ReLU, Concatenate, Conv2D, Add, Dense, MaxPool2D, AvgPool2D, Flatten, multiply, Softmax
from keras.layers import Dropout, Dense, GlobalAveragePooling2D, Input, BatchNormalization, DepthwiseConv2D, ZeroPadding2D, LayerNormalization
from tensorflow.keras import backend as K

from keras.models import Model
#import tensorflow.keras

# 정상적으로 작동 (temperature제외)

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/" "keras-applications/mobilenet/"
)


@keras_export(
    "keras.applications.mobilenet.MobileNet", "keras.applications.MobileNet"
)
def MobileNet(
    input_shape=None,
    alpha=1.0,
    depth_multiplier=1,
    dropout=1e-3,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs,
):

#     global layers
#     if "layers" in kwargs:
#         layers = kwargs.pop("layers")
#     else:
#         layers = VersionAwareLayers()
    if kwargs:
        raise ValueError(f"Unknown argument(s): {(kwargs,)}")
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  "
            f"Received weights={weights}"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top` '
            "as true, `classes` should be 1000.  "
            f"Received classes={classes}"
        )

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == "imagenet":
        if depth_multiplier != 1:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "depth multiplier must be 1.  "
                f"Received depth_multiplier={depth_multiplier}"
            )

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "alpha can be one of"
                "`0.25`, `0.50`, `0.75` or `1.0` only.  "
                f"Received alpha={alpha}"
            )

        if rows != cols or rows not in [128, 160, 192, 224]:
            rows = 224
            logging.warning(
                "`input_shape` is undefined or non-square, "
                "or `rows` is not in [128, 160, 192, 224]. "
                "Weights for input shape (224, 224) will be "
                "loaded as the default."
            )

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    num_heads = 4
    expansion_factor = 3
    
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _transformer_block(x, num_heads, expansion_factor)
        
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = _transformer_block(x, num_heads, expansion_factor)

    x = _depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _transformer_block(x, num_heads, expansion_factor)
    
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    x = _transformer_block(x, num_heads, expansion_factor)
    
    x = _depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _transformer_block(x, num_heads, expansion_factor)
    
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    x = _transformer_block(x, num_heads, expansion_factor)

    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    #x = _transformer_block(x, num_heads, expansion_factor)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    #x = _transformer_block(x, num_heads, expansion_factor)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    #x = _transformer_block(x, num_heads, expansion_factor)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    #x = _transformer_block(x, num_heads, expansion_factor)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    #x = _transformer_block(x, num_heads, expansion_factor)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    #x = _transformer_block(x, num_heads, expansion_factor)

    x = _depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    #x = _transformer_block(x, num_heads, expansion_factor)
    
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    #x = _transformer_block(x, num_heads, expansion_factor)

    if include_top:
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Dropout(dropout, name="dropout")(x)
        x = layers.Conv2D(classes, (1, 1), padding="same", name="conv_preds")(x)
        x = layers.Reshape((classes,), name="reshape_2")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Activation(
            activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name="mobilenet_%0.2f_%s" % (alpha, rows))

    # Load weights.
    if weights == "imagenet":
        if alpha == 1.0:
            alpha_text = "1_0"
        elif alpha == 0.75:
            alpha_text = "7_5"
        elif alpha == 0.50:
            alpha_text = "5_0"
        else:
            alpha_text = "2_5"

        if include_top:
            model_name = "mobilenet_%s_%d_tf.h5" % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        else:
            model_name = "mobilenet_%s_%d_tf_no_top.h5" % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights, by_name=True)

    return model


class MDTA(keras.layers.Layer):
    '''***IMPORTANT*** - The channels must be zero when divided by num_heads'''
    def __init__(self, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        #self.temperature = tf.Variable([[[[1.]] for _ in range(self.num_heads)]], shape=[None, self.num_heads, 1, 1], trainable=True)

    def build(self, inputs):
        '''(N, H, W, C) -> (N, H, W, C)
           Output of MDTA feature should be added to input feature x'''
        b, h, w, c = inputs.shape
        
        # --------------------  Layers  -------------------- 
        qkv = Conv2D(filters=c*3, kernel_size=1, use_bias=False) 
        qkv_conv = Conv2D(c*3, kernel_size=3, padding='same', groups=c*3, use_bias=False)
        project_out = Conv2D(filters=c, kernel_size=1, use_bias=False)

        temperature = tf.Variable([[[[1.]] for _ in range(self.num_heads)]], shape=[None, self.num_heads, 1, 1], trainable=True)
        
        # --------------------  forward  -------------------- 
        q, k, v = tf.split(qkv_conv(qkv(inputs)), num_or_size_splits=3, axis=-1)
        
        # divide the # of channels into heads & learn separate attention map
        q = tf.reshape(q, [-1, self.num_heads, c//self.num_heads, h * w])  # (N, num_heads, C/num_heads, HW)
        k = tf.reshape(k, [-1, self.num_heads, c//self.num_heads, h * w])
        v = tf.reshape(v, [-1, self.num_heads, c//self.num_heads, h * w])
        
        q, k = tf.nn.l2_normalize(q, axis=-1), tf.nn.l2_normalize(k, axis=-1)

        # CxC Attention map instead of HWxHW (when num_heads=1)
        attn = tf.matmul(q, k, transpose_b=True) 
        attn = multiply([attn, temperature])
        attn = Softmax(axis=-1)(attn)
        
        out = tf.matmul(attn, v)
        shape = [tf.shape(out)[k] for k in range(4)]  # [Batch, num_heads, c/num_heads, H*W]
        out = tf.reshape(out,  [shape[0], h, w, shape[1]*shape[2]])
        out = project_out(out)  # attn*v: (N, num_heads, C/num_heads, HW)
        return out
    
    def __call__(self, inputs):
        return self.build(inputs)
    

class GDFN(keras.layers.Layer):
    def __init__(self):
        super(GDFN, self).__init__()
        self.expansion_factor = 2

    def build(self, inputs):
        '''(N, H, W, C) -> (N, H, W, C) with expansion_factor=r
           Output of GDFN feature should be added to input feature x'''
        b, h, w, c = inputs.shape
        hidden_channels = int(c * self.expansion_factor)  # channel expansion 
        
        # --------------------  Layers  -------------------- 
        project_in = Conv2D(hidden_channels * 2, kernel_size=1, use_bias=False)
        conv = Conv2D(hidden_channels * 2, kernel_size=3, padding='same',
                      groups=hidden_channels * 2, use_bias=False)
        project_out = Conv2D(c, kernel_size=1, use_bias=False)
        
        # --------------------  Forward  -------------------- 
        x = project_in(inputs)  # (N, H, W, 2Cr)
        x = conv(x)  # (N, H, W, 2Cr)

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)  # (N, H, W, Cr), (N, H, W, Cr)

        # Gating: the element-wise product of 2 parallel paths of linear transformation layers 
        out = ReLU()(x1) * x2  # (N, H, W, Cr)
        out = project_out(out)  # (N, H, W, Cr)
        return out
    
    def __call__(self, inputs):
        return self.build(inputs)
    
    
def _transformer_block(inputs, num_heads, expansion_factor):
    '''(N, H, W, C) -> (N, H, W, C)'''
    
    shape = [tf.shape(inputs)[k] for k in range(4)]
    b, h, w, c = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]
    assert c % num_heads == 0   

    norm1 = LayerNormalization()  # default: axis=-1
    attn = MDTA(num_heads)
    norm2 = LayerNormalization()
    ffn = GDFN()
        
    # Add MDTA output feature
    inputs_norm1 = norm1(tf.reshape(inputs, [-1, h*w, c]))
    inputs_norm1 = tf.reshape(inputs_norm1, [-1, h, w, c])
        
    inputs = inputs + attn(inputs_norm1)
        
    # ADD GDFN output feature
    inputs_norm2 = norm2(tf.reshape(inputs, [-1, h*w, c]))
    inputs_norm2 = tf.reshape(inputs_norm2, [-1, h, w, c])
        
    x = inputs + ffn(inputs_norm2)
        
    return x
    
    
def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
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
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = ZeroPadding2D(
            ((0, 1), (0, 1)), name="conv_pad_%d" % block_id
        )(inputs)
    x = DepthwiseConv2D(
        (3, 3),
        padding="same" if strides == (1, 1) else "valid",
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name="conv_dw_%d" % block_id,
    )(x)

    x = BatchNormalization(
        axis=channel_axis, name="conv_dw_%d_bn" % block_id
    )(x)
    x = ReLU(6.0, name="conv_dw_%d_relu" % block_id)(x)

    x = Conv2D(
        pointwise_conv_filters,
        (1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1),
        name="conv_pw_%d" % block_id,
    )(x)
    x = BatchNormalization(
        axis=channel_axis, name="conv_pw_%d_bn" % block_id
    )(x)
    return ReLU(6.0, name="conv_pw_%d_relu" % block_id)(x)


def gen_mobilenetv1_mdta(input_shape, dropout_rate, num_class):
    if input_shape==(224, 224, 3):
        weights = 'imagenet'
    else:
        weights = None
        
    base_model = MobileNet(weights=weights,
                           include_top=False, 
                           input_tensor=Input(input_shape),
                           input_shape=input_shape)
    
    base_model.trainable = True
    x = base_model.output
    head_layer = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(name='simple_classifier_pooling'),
                         tf.keras.layers.Dropout(dropout_rate, name='simple_classifier_dropout'),
                         tf.keras.layers.Dense(512, activation='relu', name='simple_classifier_dense1'),
                         tf.keras.layers.Dense(num_class, activation='softmax'),])
    predictions = head_layer(x)

    # this is the model we will train
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    #print(model)
    return model

input_shape = (768, 768, 3)
x = Input(input_shape)
model = gen_mobilenetv1_mdta(input_shape, 0.3, 6)
out = model(x)

save_path = 'D:/model_mdta.h5'
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
model.save(save_path )

@keras.saving.register_keras_serializable()
class MDTA(keras.layers.Layer):
    '''***IMPORTANT*** - The channels must be zero when divided by num_heads'''
    def __init__(self, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = tf.Variable([[[[1.]] for _ in range(self.num_heads)]], shape=[None, self.num_heads, 1, 1], trainable=True)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "temperature": keras.saving.serialize_keras_object(self.temperature),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        custom_config = config.pop("temperature")
        temperature = keras.saving.deserialize_keras_object(custom_config)
        return cls(sublayer, **config)