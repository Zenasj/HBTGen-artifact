# tf.random.uniform((32, 16, 512, 1), dtype=tf.float32) ← Inferred input shape and dtype from dataset generation

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        initializer = 'he_uniform'
        nb_filts = [8, 16, 32, 400]
        out_size = 1

        # Define CNN layers as per original Sequential model
        self.conv1a = tf.keras.layers.Conv2D(nb_filts[0], kernel_size=(3, 3), activation='relu',
                                             padding='same', kernel_initializer=initializer,
                                             bias_initializer=initializer, input_shape=(16, 512, 1))
        self.conv1b = tf.keras.layers.Conv2D(nb_filts[0], kernel_size=(3, 3), activation='relu',
                                             padding='same', kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        self.conv2a = tf.keras.layers.Conv2D(nb_filts[1], kernel_size=(3, 3), activation='relu',
                                             padding='same', kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.conv2b = tf.keras.layers.Conv2D(nb_filts[1], kernel_size=(3, 3), activation='relu',
                                             padding='same', kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        self.conv3a = tf.keras.layers.Conv2D(nb_filts[2], kernel_size=(3, 3), activation='relu',
                                             padding='same', kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.conv3b = tf.keras.layers.Conv2D(nb_filts[2], kernel_size=(3, 3), activation='relu',
                                             padding='same', kernel_initializer=initializer,
                                             bias_initializer=initializer)
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu',
                                            kernel_initializer=initializer,
                                            bias_initializer=initializer)
        self.dense2 = tf.keras.layers.Dense(nb_filts[3], activation='relu',
                                            kernel_initializer=initializer,
                                            bias_initializer=initializer)
        self.out = tf.keras.layers.Dense(out_size)

    def call(self, inputs, training=False):
        x = self.conv1a(inputs)
        x = self.conv1b(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.out(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the model's expected input:
    # batch size = 32 (in original dataset), shape = (16,512,1)
    return tf.random.uniform((32, 16, 512, 1), dtype=tf.float32)

