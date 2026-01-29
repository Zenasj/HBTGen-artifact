# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← CIFAR-10 image input shape

import tensorflow as tf

class Batch_Normalization(tf.keras.layers.Layer):
    def __init__(self, depth, decay, convolution):
        super(Batch_Normalization, self).__init__()
        self.mean = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False)
        self.var = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)
        self.beta = tf.Variable(tf.constant(0.0, shape=[depth]))
        self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
        # Use tf.keras.mixed_precision.ExponentialMovingAverage instead (or track variables manually)
        # but since tf.train.ExponentialMovingAverage is a ResourceVariable, 
        # for proper tracking we wrap related variables in this layer
        
        # Using self.moving_mean and self.moving_variance as Alias for mean and var for tracking
        # We can't directly use tf.train.ExponentialMovingAverage as a Layer attribute 
        # because it tracks variables outside scope.
        self.decay = decay
        self.convolution = convolution
        self.epsilon = 0.001

    def call(self, x, training=True):
        if training:
            # Calculate batch statistics
            if self.convolution:
                batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=False)
            else:
                batch_mean, batch_var = tf.nn.moments(x, axes=[0], keepdims=False)

            # Update the moving averages manually for tracking correctness
            assign_mean = self.mean.assign(batch_mean)
            assign_var = self.var.assign(batch_var)
            with tf.control_dependencies([assign_mean, assign_var]):
                # Perform batch normalization with current batch stats
                x = tf.nn.batch_normalization(
                    x,
                    mean=batch_mean,
                    variance=batch_var,
                    offset=self.beta,
                    scale=self.gamma,
                    variance_epsilon=self.epsilon,
                )
        else:
            # Use the stored moving mean and variance at inference
            x = tf.nn.batch_normalization(
                x,
                mean=self.mean,
                variance=self.var,
                offset=self.beta,
                scale=self.gamma,
                variance_epsilon=self.epsilon,
            )
        return x


class Convolution_Layer(tf.keras.layers.Layer):
    def __init__(self, kernel_height, kernel_width, channel_in, channel_out, stride, padding):
        super(Convolution_Layer, self).__init__()
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.initializer = tf.initializers.GlorotUniform()

        # Weights are tracked by add_weight for compatibility
        self.W = self.add_weight(
            shape=(kernel_height, kernel_width, channel_in, channel_out),
            initializer=self.initializer,
            trainable=True,
            name="W",
        )
        self.b = self.add_weight(
            shape=(channel_out,),
            initializer=tf.zeros_initializer(),
            trainable=True,
            name="b",
        )

    def call(self, x):
        x = tf.nn.conv2d(x, self.W, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        x = x + self.b
        return x


class MaxPool(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides, padding):
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def call(self, x):
        return tf.nn.max_pool2d(
            x,
            ksize=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.strides, self.strides, 1],
            padding=self.padding,
        )


class Dense_layer(tf.keras.layers.Layer):
    def __init__(self, dim_out):
        super(Dense_layer, self).__init__()
        self.initializer = tf.initializers.GlorotUniform()
        self.dim_out = dim_out

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.dim_out), initializer=self.initializer, trainable=True, name="w"
        )
        self.b = self.add_weight(
            shape=(self.dim_out,), initializer=tf.zeros_initializer(), trainable=True, name="b"
        )

    def call(self, x):
        return tf.matmul(x, self.W) + self.b


class Flatten_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Flatten_layer, self).__init__()

    def call(self, x, shape=False):
        return tf.reshape(x, [tf.shape(x)[0], -1])


class Softmax_layer(tf.keras.layers.Layer):
    def __init__(self, dim_out):
        super(Softmax_layer, self).__init__()
        self.initializer = tf.initializers.GlorotUniform()
        self.dim_out = dim_out

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.dim_out),
            initializer=self.initializer,
            trainable=True,
            name="w_s",
        )
        self.b = self.add_weight(
            shape=(self.dim_out,), initializer=tf.zeros_initializer(), trainable=True, name="b_s"
        )

    def call(self, x):
        logits = tf.matmul(x, self.W) + self.b
        return tf.nn.softmax(logits)


class AvgPooling(tf.keras.layers.Layer):
    def __init__(self, kernel_height, kernel_width, strides, padding):
        super(AvgPooling, self).__init__()
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.strides = strides
        self.padding = padding

    def call(self, x):
        return tf.nn.avg_pool2d(
            x,
            ksize=[1, self.kernel_height, self.kernel_width, 1],
            strides=[1, self.strides, self.strides, 1],
            padding=self.padding,
        )


class Dropout_layer(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(Dropout_layer, self).__init__()
        self.rate = rate

    def call(self, x, training=None):
        if training:
            return tf.nn.dropout(x, rate=self.rate)
        return x


class Identity(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Identity, self).__init__()
        self.initializer = tf.initializers.GlorotUniform()

        # Block one
        self.conv_1 = Convolution_Layer(1, 1, filters[0], filters[1], 1, padding="VALID")
        self.batch_norm_1 = Batch_Normalization(filters[1], 0.99, convolution=True)

        # Block two
        self.conv_2 = Convolution_Layer(3, 3, filters[1], filters[2], 1, padding="SAME")
        self.batch_norm_2 = Batch_Normalization(filters[2], 0.99, convolution=True)

        # Block three
        self.conv_3 = Convolution_Layer(1, 1, filters[2], filters[3], 1, padding="VALID")
        self.batch_norm_3 = Batch_Normalization(filters[3], 0.99, convolution=True)

    def call(self, x, training=None):
        # Block one
        fx = self.conv_1(x)
        fx = self.batch_norm_1(fx, training)
        fx = tf.nn.relu(fx)

        # Block two
        fx = self.conv_2(fx)
        fx = self.batch_norm_2(fx, training)
        fx = tf.nn.relu(fx)

        # Block three
        fx = self.conv_3(fx)
        fx = self.batch_norm_3(fx, training)

        # Add input (identity connection)
        fx = tf.nn.relu(fx + x)

        return fx


class Convolution_Block(tf.keras.layers.Layer):
    def __init__(self, filters, stride):
        super(Convolution_Block, self).__init__()
        self.initializer = tf.initializers.GlorotUniform()

        # Block one
        self.conv_1 = Convolution_Layer(1, 1, filters[0], filters[1], stride, padding="VALID")
        self.batch_norm_1 = Batch_Normalization(filters[1], 0.99, convolution=True)

        # Block two
        self.conv_2 = Convolution_Layer(3, 3, filters[1], filters[2], 1, padding="SAME")
        self.batch_norm_2 = Batch_Normalization(filters[2], 0.99, convolution=True)

        # Block three
        self.conv_3 = Convolution_Layer(1, 1, filters[2], filters[3], 1, padding="VALID")
        self.batch_norm_3 = Batch_Normalization(filters[3], 0.99, convolution=True)

        # Dimension adjustment variable (skip connection conv)
        self.dimension = Convolution_Layer(1, 1, filters[0], filters[3], stride, padding="VALID")

    def call(self, x, training=None):
        # Block one
        fx = self.conv_1(x)
        fx = self.batch_norm_1(fx, training)
        fx = tf.nn.relu(fx)

        # Block two
        fx = self.conv_2(fx)
        fx = self.batch_norm_2(fx, training)
        fx = tf.nn.relu(fx)

        # Block three
        fx = self.conv_3(fx)
        fx = self.batch_norm_3(fx, training)

        # Skip connection
        fx = tf.nn.relu(fx + self.dimension(x))
        return fx


class Global_Average_Pooling(tf.keras.layers.Layer):
    def __init__(self, axis):
        super(Global_Average_Pooling, self).__init__()
        self.axis = axis

    def call(self, x):
        return tf.reduce_mean(x, axis=self.axis)


class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()

        # Input image assumed shape (B, 32, 32, 3)
        self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=(3, 3))
        self.conv_1 = Convolution_Layer(
            kernel_height=7, kernel_width=7, channel_in=3, channel_out=64, stride=2, padding="VALID"
        )
        self.batch_norm_1 = Batch_Normalization(depth=64, decay=0.99, convolution=True)
        self.max_pool_1 = MaxPool(kernel_size=3, strides=2, padding="VALID")

        # 2nd stage
        self.block_2_res_conv_1 = Convolution_Block(filters=[64, 64, 64, 256], stride=1)
        self.block_2_res_ident_1 = Identity(filters=[256, 64, 64, 256])
        self.block_2_res_ident_2 = Identity(filters=[256, 64, 64, 256])

        # 3rd stage
        self.block_3_res_conv_1 = Convolution_Block(filters=[256, 128, 128, 512], stride=2)
        self.block_3_res_ident_1 = Identity(filters=[512, 128, 128, 512])
        self.block_3_res_ident_2 = Identity(filters=[512, 128, 128, 512])
        self.block_3_res_ident_3 = Identity(filters=[512, 128, 128, 512])

        # 4th stage
        self.block_4_res_conv_1 = Convolution_Block(filters=[512, 256, 256, 1024], stride=2)
        self.block_4_res_ident_1 = Identity(filters=[1024, 256, 256, 1024])
        self.block_4_res_ident_2 = Identity(filters=[1024, 256, 256, 1024])
        self.block_4_res_ident_3 = Identity(filters=[1024, 256, 256, 1024])
        self.block_4_res_ident_4 = Identity(filters=[1024, 256, 256, 1024])
        self.block_4_res_ident_5 = Identity(filters=[1024, 256, 256, 1024])

        # 5th stage
        self.block_5_res_conv_1 = Convolution_Block(filters=[1024, 512, 512, 2048], stride=2)
        self.block_5_res_ident_1 = Identity(filters=[1024, 512, 512, 2048])
        self.block_5_res_ident_2 = Identity(filters=[1024, 512, 512, 2048])

        self.global_avg_pool = Global_Average_Pooling(axis=[1, 2])
        self.dense_1 = Dense_layer(512)
        self.softmax = Softmax_layer(num_classes)

    def call(self, x, training=None):
        x = self.zero_padding(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x, training)
        x = tf.nn.relu(x)
        x = self.max_pool_1(x)

        # 2nd stage
        x = self.block_2_res_conv_1(x, training)
        x = self.block_2_res_ident_1(x, training)
        x = self.block_2_res_ident_2(x, training)

        # 3rd stage
        x = self.block_3_res_conv_1(x, training)
        x = self.block_3_res_ident_1(x, training)
        x = self.block_3_res_ident_2(x, training)
        x = self.block_3_res_ident_3(x, training)

        # 4th stage
        x = self.block_4_res_conv_1(x, training)
        x = self.block_4_res_ident_1(x, training)
        x = self.block_4_res_ident_2(x, training)
        x = self.block_4_res_ident_3(x, training)
        x = self.block_4_res_ident_4(x, training)
        x = self.block_4_res_ident_5(x, training)

        # 5th stage
        x = self.block_5_res_conv_1(x, training)
        x = self.block_5_res_ident_1(x, training)
        x = self.block_5_res_ident_2(x, training)

        x = self.global_avg_pool(x)
        x = self.dense_1(x)
        x = tf.nn.relu(x)
        x = self.softmax(x)

        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel(num_classes=10)


def GetInput():
    # CIFAR-10 images are 32x32 RGB images, dtype float32 typically normalized [0,1].
    # Return a batch of 1 random image (batch size 1).
    return tf.random.uniform(shape=(1, 32, 32, 3), minval=0, maxval=1, dtype=tf.float32)

