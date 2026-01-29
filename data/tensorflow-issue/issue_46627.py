# tf.random.uniform((None, 28, 28), dtype=tf.float32) ← Input shape is (batch_size, 28, 28) grayscale images like MNIST digits

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances."""
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < 0.5, y_true.dtype)), tf.float32))

class BaseNetwork(Model):
    def __init__(self, input_shape):
        super(BaseNetwork, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.6)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.1)
        self.dense3 = layers.Dense(128, activation='relu')

    def call(self, inputs, training=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        return x

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(28, 28)):
        super(MyModel, self).__init__()
        # Shared base network for feature extraction
        self.base_network = BaseNetwork(input_shape)
        self.input_shape_ = input_shape

    @tf.function
    def call(self, inputs, training=None):
        """
        Inputs: a tuple or list of two tensors: (input_a, input_b)
          where each input shape is (batch_size, 28, 28)
        Outputs:
          - distance tensor of shape (batch_size, 1) representing Euclidean distance between embeddings
        """
        input_a, input_b = inputs
        processed_a = self.base_network(input_a, training=training)
        processed_b = self.base_network(input_b, training=training)
        distance = euclidean_distance((processed_a, processed_b))
        return distance

def my_model_function():
    # Create an instance of MyModel with MNIST image input shape (28,28)
    return MyModel(input_shape=(28, 28))

def GetInput():
    # Return a tuple of two random image batches corresponding to input pairs
    # Batch size chosen arbitrarily as 4 for demonstration (can be any batch size)
    batch_size = 4
    # Input images are grayscale 28x28 float32 in [0,1]
    input_a = tf.random.uniform((batch_size, 28, 28), minval=0.0, maxval=1.0, dtype=tf.float32)
    input_b = tf.random.uniform((batch_size, 28, 28), minval=0.0, maxval=1.0, dtype=tf.float32)
    return (input_a, input_b)

