# tf.random.uniform((B, 40, 200), dtype=tf.float32)  # Input shape inferred from batch generator: (batch_size, features=40, frames=200)

import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int = None,
        output_dim: int = 512,
        context_dim: int = 5,
        stride: int = 1,
        dilation: int = 1,
        kernel_initializer='glorot_uniform'
    ):
        super(MyLayer, self).__init__()
        self.input_dim = input_dim
        self.conv = tf.keras.layers.Conv1D(filters=output_dim,
                                           kernel_size=context_dim,
                                           strides=stride,
                                           dilation_rate=dilation,
                                           kernel_initializer=kernel_initializer,
                                           padding='valid')  # padding assumed default 'valid'

    def call(self, x):
        # Shapes inferred but not used explicitly; compatible with Keras layers
        return self.conv(x)

class Reg(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.2, batch_norm=False):
        super(Reg, self).__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = tf.keras.layers.BatchNormalization()
        self.do = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, x, training=True):
        if self.batch_norm:
            x = self.bn(x, training=training)
        return self.do(x, training=training)

class MyModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim,
                 dropout_rate=0.2,
                 batch_norm=False,
                 return_xvector=False):
        super().__init__()

        self.input_dim = input_dim  # number of features: 40
        self.output_dim = output_dim  # usually number of classes: 3

        # Five 1D convolutional layers with increasing dilation rates and filters
        self.fc1 = MyLayer(input_dim=self.input_dim, output_dim=512, context_dim=5, dilation=1)
        self.fc2 = MyLayer(input_dim=512, output_dim=1536, context_dim=3, dilation=2)
        self.fc3 = MyLayer(input_dim=1536, output_dim=512, context_dim=3, dilation=3)
        self.fc4 = MyLayer(input_dim=512, output_dim=512, context_dim=1, dilation=1)
        self.fc5 = MyLayer(input_dim=512, output_dim=1500, context_dim=1, dilation=1)
        
        # Fully connected layers after stats pooling
        self.fc6 = tf.keras.layers.Dense(512)
        self.fc7 = tf.keras.layers.Dense(512)
        self.output_layer = tf.keras.layers.Dense(self.output_dim)

        # Choose BatchNorm + Dropout or Dropout only for regularization layers
        self.reg1 = Reg(dropout_rate, batch_norm=batch_norm)
        self.reg2 = Reg(dropout_rate, batch_norm=batch_norm)
        self.reg3 = Reg(dropout_rate, batch_norm=batch_norm)
        self.reg4 = Reg(dropout_rate, batch_norm=batch_norm)
        self.reg5 = Reg(dropout_rate, batch_norm=batch_norm)
        self.reg6 = Reg(dropout_rate, batch_norm=batch_norm)

        # Softmax for final probabilities (used if return_logits=False)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

        # Support for optionally returning x-vector embeddings (not used here)
        self.return_xvector = return_xvector

    def call(self, x, training=True, return_logits=True):
        # Extractor stack with conv layers + relu + dropout/batchnorm
        with tf.name_scope("Extractor"):
            with tf.name_scope("Fc1"):
                x = tf.nn.relu(self.fc1(x))
                x = self.reg1(x, training=training)
            with tf.name_scope("Fc2"):
                x = tf.nn.relu(self.fc2(x))
                x = self.reg2(x, training=training)
            with tf.name_scope("Fc3"):
                x = tf.nn.relu(self.fc3(x))
                x = self.reg3(x, training=training)
            with tf.name_scope("Fc4"):
                x = tf.nn.relu(self.fc4(x))
                x = self.reg4(x, training=training)
            with tf.name_scope("Fc5"):
                x = tf.nn.relu(self.fc5(x))
                x = self.reg5(x, training=training)
            with tf.name_scope("StatsPool"):
                x = self.statpool(x)

            with tf.name_scope("Segment6"):
                x = self.fc6(x)

        # Classifier fully connected layers + dropout/batchnorm + final logits
        with tf.name_scope("Classifier"):
            x = tf.nn.relu(x)
            x = self.reg6(x, training=training)
            x = tf.nn.relu(self.fc7(x))
            x = self.output_layer(x)
        
        if return_logits:
            return x
        else:
            return self.softmax(x)

    def statpool(self, x):
        # Statistics pooling: concatenate mean and stddev over time dimension (axis=1)
        mu = tf.math.reduce_mean(x, axis=1)
        sigma = tf.math.reduce_std(x, axis=1)
        return tf.concat([mu, sigma], axis=1)

def my_model_function():
    # Return an instance of MyModel with input_dim=40 features, output_dim=3 classes
    # Using default dropout 0.2, no batch norm by default to replicate original
    return MyModel(input_dim=40, output_dim=3, dropout_rate=0.2, batch_norm=False)

def GetInput():
    # Return random input tensor shaped (batch_size, features=40, time_steps=200)
    # Matching the original batch generator shape: (N, n_feats, 200)
    # Using uniform distribution instead of normal for variety; dtype float32
    batch_size = 4  # small batch size for example; can be any positive integer
    n_feats = 40
    time_steps = 200
    return tf.random.uniform(shape=(batch_size, n_feats, time_steps), dtype=tf.float32)

