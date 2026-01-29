# tf.random.uniform((B, 128, 128, 3), dtype=tf.float32)  # Assumption: input images 128x128 RGB as input_shape is not explicitly given

import tensorflow as tf
from tensorflow.keras import layers, Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Based on the issue model architecture:
        # Input assumed to be (128,128,3) since shape is not specified in the issue.
        # This matches conv2d first layer output shape noted in logs.
        self.conv1 = layers.Conv2D(32, (3,3), padding="same", activation='relu')
        self.conv2 = layers.Conv2D(32, (3,3), activation='relu')
        self.pool1 = layers.MaxPooling2D(pool_size=(2,2))
        self.dropout1 = layers.Dropout(0.5)

        self.conv3 = layers.Conv2D(64, (3,3), activation='relu')
        self.pool2 = layers.MaxPooling2D(pool_size=(2,2))
        self.dropout2 = layers.Dropout(0.5)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout3 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(34, activation='softmax')  # num_classes=34 as per the original code

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Creates and returns an instance of MyModel
    model = MyModel()
    # Compile with SGD and categorical_crossentropy and accuracy following original issue
    # Optimizer and loss as per original; metrics accuracy
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a random input tensor with shape (batch_size, 128, 128, 3)
    # Assumed 3 channels (RGB) as input_shape was not specified in original issue.
    # Batch size chosen as 32, matching batch size in fit() from issue.
    batch_size = 32
    input_shape = (batch_size, 128, 128, 3)  
    return tf.random.uniform(input_shape, dtype=tf.float32)

