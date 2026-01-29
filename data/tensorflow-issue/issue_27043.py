# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST example with grayscale images

import tensorflow as tf

EPS = 1e-5
MOMENTUM = 0.9

def pool2d_layer(inputs, pool_type, pool_size=2, pool_stride=2):
    if pool_type == "max":
        # Max pooling layer
        return tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride)(inputs)
    elif pool_type == "average":
        return tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_stride)(inputs)
    else:
        return inputs  # no pooling if unknown type

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Config parameters inferred from original code
        self.num_classes = 10
        self.filter_sizes_conv_layers = [(5, 32), (5, 64)]
        self.num_units_fc_layers = [512]
        self.pool_params = {"type": "max", "size": 2, "stride": 2}
        self.dropout_rate = 0.0
        self.batch_norm = True
        self.activation = tf.nn.relu
        
        # Build convolutional layers with batch norm layers
        self.conv_layers = []
        self.bn_layers = []
        in_channel = 1  # grayscale input channel
        
        # Using tf.keras.layers.Conv2D to be consistent with tf.keras.Model
        for (kernel_size, filters) in self.filter_sizes_conv_layers:
            conv_layer = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                activation=self.activation if not self.batch_norm else None,
                use_bias=not self.batch_norm
            )
            self.conv_layers.append(conv_layer)
            bn_layer = tf.keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPS)
            self.bn_layers.append(bn_layer)
            in_channel = filters
        
        # Fully connected layers
        self.fc_layers = []
        for units in self.num_units_fc_layers:
            fc_layer = tf.keras.layers.Dense(units, activation=self.activation)
            self.fc_layers.append(fc_layer)
        
        # Output layer without activation
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation=None)
        
        # Dropout layer, only used if dropout_rate > 0
        if self.dropout_rate > 0:
            self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)
        else:
            self.dropout_layer = None

    def call(self, inputs, training=False):
        x = inputs
        for i in range(len(self.filter_sizes_conv_layers)):
            x = self.conv_layers[i](x)
            if self.batch_norm:
                x = self.bn_layers[i](x, training=training)
            if self.pool_params:
                x = pool2d_layer(
                    x,
                    pool_type=self.pool_params["type"],
                    pool_size=self.pool_params["size"],
                    pool_stride=self.pool_params["stride"]
                )
            if self.dropout_layer is not None:
                x = self.dropout_layer(x, training=training)
        
        x = tf.keras.layers.Flatten()(x)
        
        for fc in self.fc_layers:
            x = fc(x)
        
        logits = self.output_layer(x)
        return logits

def my_model_function():
    """
    Returns an instance of MyModel, initialized with batch_norm enabled 
    as in the original example.
    """
    model = MyModel()
    return model

def GetInput():
    """
    Returns a random tensor input matching the expected input shape for MyModel:
    (batch_size=3, height=28, width=28, channels=1), dtype float32.
    The batch size 3 is from original example.
    """
    return tf.random.uniform((3, 28, 28, 1), dtype=tf.float32)

