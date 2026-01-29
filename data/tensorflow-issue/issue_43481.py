# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← Input shape from model input (batch size 1 assumed)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers according to the original Sequential model:
        # Conv2D(32, (3,3), padding='same', input_shape=(224,224,3))
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
        self.activation = tf.keras.layers.Activation('relu')
        self.maxpool = tf.keras.layers.MaxPooling2D((2, 2))
        # As of TF 2.2.0, UpSampling2D interpolation='nearest' is unsupported in TFLite.
        # To maintain compatibility with TFLite, we'll omit 'interpolation' argument,
        # which defaults to 'nearest'.
        self.upsample = tf.keras.layers.UpSampling2D((2, 2))  
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.upsample(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of the MyModel class.
    return MyModel()

def GetInput():
    # Generate a random tensor input compatible with MyModel:
    # Shape: (batch_size=1, height=224, width=224, channels=3), dtype float32
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

