# tf.random.uniform((32, 32, 32, 64), dtype=tf.float32)
import tensorflow as tf

class EinSumLayer(tf.keras.layers.Layer):
    def __init__(self, name='', **kwargs):
        super(EinSumLayer, self).__init__(**kwargs)
        self._name = name

    def call(self, inputs):
        # Inputs: tuple of two tensors (both expected to be 2D) for einsum operation
        return tf.einsum('ij,jk->ik', inputs[0], inputs[1])


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        batch_size = 32
        H = 32
        W = 32
        C = 64
        input_shape = (batch_size, H, W, C)

        # Convolution and batch norm blocks with ReLU activations
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(1, 1), strides=(1, 1),
            padding='same', data_format='channels_last',
            activation=None, use_bias=True,
            input_shape=input_shape[1:])
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), strides=(1, 1),
            padding='same', data_format='channels_last',
            activation=None, use_bias=True)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(1, 1), strides=(1, 1),
            padding='same', data_format='channels_last',
            activation=None, use_bias=True)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

        self.add = tf.keras.layers.Add()

        # Max pooling and ReLU
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same',
                                               data_format='channels_last')
        self.relu4 = tf.keras.layers.ReLU()

        self.flatten = tf.keras.layers.Flatten(data_format='channels_last')

        # Dense layers with no activation (for potential fusion with BiasAdd)
        self.dense1 = tf.keras.layers.Dense(64, activation=None, use_bias=True)
        self.dense3 = tf.keras.layers.Dense(32, activation=None, use_bias=True)

        self.ein_sum = EinSumLayer('ein_sum')
        self.relu5 = tf.keras.layers.ReLU()

        # Final dense layer with ReLU activation
        self.dense_out = tf.keras.layers.Dense(1, activation='relu', use_bias=True)

        # Optimizer (if needed)
        self.optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def call(self, x, training=False):
        # Save input tensor for residual connection
        y = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)

        # Residual add
        x = self.add([x, y])

        x = self.pool1(x)
        x = self.relu4(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense3(x)

        # ein_sum expects tuple of two tensors; passing (x, x)
        x = self.ein_sum((x, x))
        x = self.relu5(x)

        x = self.dense_out(x)

        return x


def my_model_function():
    # Return an initialized instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor matching input expected by MyModel
    # Shape: (batch_size=32, H=32, W=32, C=64), dtype=tf.float32
    return tf.random.uniform(shape=(32, 32, 32, 64), dtype=tf.float32)

