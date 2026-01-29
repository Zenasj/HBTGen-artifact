# tf.random.uniform((BATCH_SIZE, 8, 32, 32, 3), dtype=tf.float32) ← Input shape inferred from create_model input_shape parameter

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        act = "relu"
        pad = "same"
        ini = "he_uniform"

        # Replicating the same Conv3D layers and structure from the Sequential model
        self.conv3d_1 = tf.keras.layers.Conv3D(128, (3,3,3), activation=act, padding=pad,
                                               kernel_initializer=ini, input_shape=(8,32,32,3))
        self.conv3d_2 = tf.keras.layers.Conv3D(256, (3,3,3), activation=act, padding=pad,
                                               kernel_initializer=ini)
        self.conv3d_3 = tf.keras.layers.Conv3D(256, (3,3,3), activation=act, padding=pad,
                                               kernel_initializer=ini)
        self.conv3d_4 = tf.keras.layers.Conv3D(256, (3,3,3), activation=act, padding=pad,
                                               kernel_initializer=ini)
        self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2))

        self.conv3d_5 = tf.keras.layers.Conv3D(256, (3,3,3), activation=act, padding=pad,
                                               kernel_initializer=ini)
        self.conv3d_6 = tf.keras.layers.Conv3D(256, (3,3,3), activation=act, padding=pad,
                                               kernel_initializer=ini)
        self.conv3d_7 = tf.keras.layers.Conv3D(512, (3,3,3), activation=act, padding=pad,
                                               kernel_initializer=ini)
        self.conv3d_8 = tf.keras.layers.Conv3D(512, (3,3,3), activation=act, padding=pad,
                                               kernel_initializer=ini)
        self.pool2 = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2))

        self.conv3d_9 = tf.keras.layers.Conv3D(256, (3,3,3), activation=act, padding=pad,
                                               kernel_initializer=ini)
        self.conv3d_10 = tf.keras.layers.Conv3D(256, (3,3,3), activation=act, padding=pad,
                                                kernel_initializer=ini)
        self.conv3d_11 = tf.keras.layers.Conv3D(256, (3,3,3), activation=act, padding=pad,
                                                kernel_initializer=ini)
        self.conv3d_12 = tf.keras.layers.Conv3D(128, (3,3,3), activation=act, padding=pad,
                                                kernel_initializer=ini)
        self.pool3 = tf.keras.layers.MaxPooling3D(pool_size=(2,4,4))

        self.flatten = tf.keras.layers.Flatten()
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv3d_1(inputs)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv3d_4(x)
        x = self.pool1(x)

        x = self.conv3d_5(x)
        x = self.conv3d_6(x)
        x = self.conv3d_7(x)
        x = self.conv3d_8(x)
        x = self.pool2(x)

        x = self.conv3d_9(x)
        x = self.conv3d_10(x)
        x = self.conv3d_11(x)
        x = self.conv3d_12(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.batchnorm(x, training=training)
        x = self.dense1(x)
        out = self.dense2(x)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random float32 tensor matching input shape (batch_size, 8, 32, 32, 3)
    # Using batch size 64 as recommended in the comments for GPU memory constraints
    batch_size = 64
    input_shape = (batch_size, 8, 32, 32, 3)
    # Normalized values roughly between 0 and 1 as per the example normalization
    return tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)

