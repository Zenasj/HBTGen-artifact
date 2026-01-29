# tf.random.uniform((None, 28, 28, 1), dtype=tf.float32) ← Inferring input shape from MNIST example with BatchNorm usage

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # Use He normal initializer like original code's he_normal
        self.init_op = tf.keras.initializers.HeNormal()
        
        # Define the layers similar to the MNIST "custom_model" example,
        # including Conv2D, ReLU activations, MaxPooling2D, Dense and Softmax.
        # Batch normalization layers are NOT included here explicitly since the issue
        # relates to their behavior in TF 2.0 with disabled eager, but we keep the logic simple,
        # so if someone wants to test BatchNorm, they can insert them.
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), padding="same",
                                            activation="relu", kernel_initializer=self.init_op, name="clf_c1")
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), padding="same",
                                            activation="relu", kernel_initializer=self.init_op, name="clf_c2")
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="clf_p1")
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), padding="same",
                                            activation="relu", kernel_initializer=self.init_op, name="clf_c3")
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), padding="same",
                                            activation="relu", kernel_initializer=self.init_op, name="clf_c4")
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="clf_p2")
        self.flatten = tf.keras.layers.Flatten(name="clf_f1")
        self.dense1 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer=self.init_op, name="clf_d1")
        self.dense2 = tf.keras.layers.Dense(10, activation=None, kernel_initializer=self.init_op, name="clf_d2")
        self.softmax = tf.keras.layers.Activation('softmax', name="clf_a1")
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor matching the expected input shape for MyModel
    # The original MNIST example expects input shape (28,28,1) and batch dimension assumed
    # Use batch size 32 for example
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

