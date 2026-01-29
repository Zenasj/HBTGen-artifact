# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← CIFAR-10 image input shape (batch, height, width, channels)

import tensorflow as tf
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate the architecture described in the issue:
        # Sequential model with Conv2D, Activation, Pooling, Dropout, Dense layers

        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3))
        self.act1 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.drop1 = tf.keras.layers.Dropout(0.25)

        self.conv3 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.drop2 = tf.keras.layers.Dropout(0.25)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.drop3 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        if training:
            x = self.drop1(x, training=training)
        else:
            x = self.drop1(x)  # no dropout if not training

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        if training:
            x = self.drop2(x, training=training)
        else:
            x = self.drop2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.drop3(x, training=training)
        else:
            x = self.drop3(x)

        x = self.dense2(x)
        return x

def loss_func(y_true, y_pred):
    # Loss function as provided in the original issue, with batch flattening
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=1, keepdims=True)
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True)
    return - (2. * intersection + K.epsilon()) / (union + K.epsilon())

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile with RMSprop optimizer and the external loss function
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss=loss_func, optimizer=opt, metrics=[])
    return model

def GetInput():
    # Return a random float32 tensor shaped (batch_size, 32, 32, 3), matching CIFAR-10 images
    # Default batch size 32 (common for training)
    return tf.random.uniform((32, 32, 32, 3), dtype=tf.float32, minval=0., maxval=1.)

