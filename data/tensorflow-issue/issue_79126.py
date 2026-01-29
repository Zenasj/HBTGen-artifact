# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← Input shape (batch size 1 assumed)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adamax

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original Sequential model architecture from the issue

        # Block 1
        self.conv1_1 = Conv2D(46, kernel_size=(3,3), activation='relu', padding='valid')
        self.conv1_2 = Conv2D(46, kernel_size=(3,3), activation='relu', padding='valid')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D((2,2))
        self.dropout1 = Dropout(0.15)

        # Block 2
        self.conv2_1 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid')
        self.conv2_2 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid')
        self.conv2_3 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D((2,2))
        self.dropout2 = Dropout(0.3)

        # Block 3
        self.conv3_1 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid')
        self.conv3_2 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid')
        self.bn3 = BatchNormalization()
        self.pool3 = MaxPooling2D((2,2))
        self.dropout3 = Dropout(0.5)

        # Block 4
        self.conv4_1 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='valid')
        self.conv4_2 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='valid')
        self.conv4_3 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='valid')
        self.bn4 = BatchNormalization()
        self.pool4 = MaxPooling2D((2,2))
        self.dropout4 = Dropout(0.1)

        # Classifier
        self.flatten = Flatten()
        self.dense = Dense(15, activation='softmax', kernel_regularizer=l2(0.16))

    def call(self, inputs, training=False):
        # Forward pass replicates the Sequential model's order
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        x = self.dropout3(x, training=training)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.bn4(x, training=training)
        x = self.pool4(x)
        x = self.dropout4(x, training=training)

        x = self.flatten(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Instantiate the model and compile it similarly to the original example
    model = MyModel()
    model.compile(
        optimizer=Adamax(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Create a batch of input images with shape (1, 224, 224, 3) matching the model's expected input
    # Batch size = 1 assumed here for simplicity; can be adjusted as needed
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

