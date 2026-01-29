# tf.random.uniform((B, 28, 28, 1), dtype=tf.float64)
import tensorflow as tf

feature_type = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the same model architecture as in the issue for MNIST-like data
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation='tanh')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='tanh')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(84, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        # Inputs expected to be (batch_size, 28, 28, 1) with dtype float64
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        return out

def my_model_function():
    # Return a new instance of MyModel
    return MyModel()

def parser_tfrecord(tfrecord_example):
    """
    Parses a single tfrecord example, decodes the serialized tensor,
    fixes its shape to (28,28,1), and returns image,label tuple.
    """
    _feature = tf.io.parse_single_example(tfrecord_example, feature_type)
    # Parsing the serialized tensor, dtype float64 as in the issue
    img = tf.io.parse_tensor(_feature['image'], out_type=tf.float64)
    # Important: Set static shape so model training recognizes shape correctly
    img.set_shape((28, 28, 1))
    label = _feature['label']
    return img, label

def GetInput():
    # Generate a random input tensor matching expected input for MyModel
    # batch size = 8 (arbitrary)
    batch_size = 8
    # Use float64 to imitate dtype parsing in issue, shape (8,28,28,1)
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float64)

