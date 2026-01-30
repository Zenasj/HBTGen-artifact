import random
from tensorflow import keras
from tensorflow.keras import layers

import os
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ""

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.enable_eager_execution(tf_config)

EPS = 1e-5
MOMENTUM = 0.9

class ConfigDict(object):
    def __init__(self):
        self.num_classes = 10

        # List of tuples specify (kernel_size, number of filters) for each layer.
        self.filter_sizes_conv_layers = [(5, 32), (5, 64)]
        # Dictionary of pooling type ("max"/"average", size and stride).
        self.pool_params = {"type": "max", "size": 2, "stride": 2}
        self.num_units_fc_layers = [512]
        self.dropout_rate = 0
        self.batch_norm = True
        self.activation = tf.nn.relu
    
def pool2d_layer(inputs, pool_type, pool_size=2, pool_stride=2):
    if pool_type == "max":
        # Max pooling layer
        return tf.layers.max_pooling2d(
            inputs, pool_size=[pool_size] * 2, strides=pool_stride)
    
    
class MNISTNetwork(tf.keras.Model):
    """MNIST model. """

    def __init__(self, config):
        super(MNISTNetwork, self).__init__()
        self.num_classes = config.num_classes
        self.var_list = []
        self.init_ops = None
        self.activation = config.activation
        self.filter_sizes_conv_layers = config.filter_sizes_conv_layers
        self.num_units_fc_layers = config.num_units_fc_layers
        self.pool_params = config.pool_params
        self.dropout_rate = config.dropout_rate
        self.batch_norm = config.batch_norm
        self.conv_layers = []
        self.bn_layers = []
        in_channel = 1
        for i, filter_size in enumerate(self.filter_sizes_conv_layers):
            f_size = filter_size[0]
            conv_layer = tf.layers.Conv2D(kernel_size=filter_size[0], filters=filter_size[1], 
                                          strides=(1, 1), padding="same",
                                          activation=self.activation, 
                                          use_bias=not self.batch_norm)
            self.conv_layers.append(conv_layer)
            batch_norm_layer = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPS)
            self.bn_layers.append(batch_norm_layer)
            in_channel = filter_size[1]
            
        self.fc_layers = []
        in_shape = 64 * 7 * 7
        for i, num_units in enumerate(self.num_units_fc_layers):
            fc_layer = tf.layers.Dense(num_units, activation=self.activation)
            self.fc_layers.append(fc_layer)
            in_shape = num_units
        self.output_layer = tf.layers.Dense(self.num_classes, activation=None)

    def __call__(self, images, is_training=False):
        """Builds model."""
        net = images
        for i in range(len(self.filter_sizes_conv_layers)):
            net = self.conv_layers[i](net)

            if self.pool_params:
                net = pool2d_layer(net, pool_type=self.pool_params["type"], pool_size=self.pool_params["size"]
                                   , pool_stride=self.pool_params["stride"])
            if self.dropout_rate > 0:
                net = tf.layers.dropout(net, rate=self.dropout_rate, training=is_training)
                
            if self.batch_norm:
                # net = tf.layers.batch_normalization(net, training=is_training, epsilon=EPS, momentum=MOMENTUM)
                net = self.bn_layers[i](net, training=is_training)
            
        net = tf.layers.flatten(net)

        for i in range(len(self.num_units_fc_layers)):
            net = self.fc_layers[i](net)
        logits = self.output_layer(net)
        return logits
    

config = ConfigDict()

# enable/disable batch norm
config.batch_norm = True

model = MNISTNetwork(config)

images = np.random.uniform(0, 1, (3, 28, 28, 1))
images = tf.convert_to_tensor(images, dtype=np.float32)
# images = tf.Variable(images)
print("data %.5f" % images.numpy().sum())

with tf.GradientTape(persistent=True) as t:
    with tf.GradientTape(persistent=True) as t2:
        logits = model(images, is_training=True)
        m = tf.reduce_sum(logits)
        print(logits.numpy().sum())
        dp_dx = t2.gradient(m, model.variables)
    print("first", dp_dx[0].numpy().sum())
    d2y_dx2 = t.gradient(dp_dx[0], model.variables)
    print("second order", d2y_dx2[0].numpy().sum())