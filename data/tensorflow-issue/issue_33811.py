# tf.random.uniform((BATCH_SIZE, 150, 150, 3), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Replicating the Sequential CNN model used in the issue
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D(2, 2)

        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2, 2)

        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(2, 2)

        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(2, 2)

        self.dropout = tf.keras.layers.Dropout(0.5)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='softmax')  # 2 classes for dogs vs cats

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile with the same configuration from the issue
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Generate random input tensor matching shape (BATCH_SIZE, 150, 150, 3)
    # Using float32 and range [0,1] as the ImageDataGenerator rescales by 1./255
    BATCH_SIZE = 100
    IMG_SHAPE = 150
    return tf.random.uniform((BATCH_SIZE, IMG_SHAPE, IMG_SHAPE, 3), dtype=tf.float32)

