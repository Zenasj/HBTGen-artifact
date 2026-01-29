# tf.random.uniform((B, 80, 320, 1), dtype=tf.float32)  ← Based on the original model input shape (80, 320, 1) grayscale image

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Rescaling, Conv2D, MaxPooling2D, Dense, Dropout,
)
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Input shape: (80, 320, 1)
        
        # Rescaling layer normalizes input images from [0, 255] to [-1, 1]
        self.rescale = Rescaling(scale=1./127.5, offset=-1.)
        
        # Convolutional and pooling layers (same architecture as v2Model)
        self.conv1 = Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='elu')
        self.pool1 = MaxPooling2D(pool_size=(2, 2), padding='valid')
        
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='elu')
        self.pool2 = MaxPooling2D(pool_size=(2, 2), padding='valid')
        
        self.conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='elu')
        self.pool3 = MaxPooling2D(pool_size=(2, 2), padding='valid')
        
        # The original code used Dense(19456) directly on conv output which is invalid:
        # We need to flatten before Dense layer. Since no Flatten was in snippet, we add it.
        self.flatten = tf.keras.layers.Flatten()
        
        self.dense1 = Dense(19456, use_bias=True)
        self.dropout1 = Dropout(rate=0.2)
        self.dense2 = Dense(500, use_bias=True)
        self.dropout2 = Dropout(rate=0.2)

        # Two separate output heads
        self.head_steering = Dense(1, activation="linear", name="output_ster")
        self.head_speed = Dense(1, activation="linear", name="output_acc")

    def call(self, inputs, training=False):
        x = self.rescale(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x, training=training)

        out_steering = self.head_steering(x)
        out_speed = self.head_speed(x)

        return {"output_ster": out_steering, "output_acc": out_speed}


def my_model_function():
    # Return an instance of MyModel.
    # Note: weights are randomly initialized here.
    model = MyModel()
    # Compile model with MeanSquaredError loss and Adam optimizer
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    # Compile to allow training without error
    # Since multiple outputs, pass a dict of losses
    model.compile(
        optimizer=optimizer,
        loss={"output_ster": loss, "output_acc": loss},
        metrics=['accuracy']
    )
    return model


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel,
    # i.e. shape (B, 80, 320, 1) with dtype float32.
    # Batch size 4 is arbitrary for demonstration.
    batch_size = 4
    # Using random uniform values to simulate image input [0,255]
    x = tf.random.uniform((batch_size, 80, 320, 1), minval=0, maxval=255, dtype=tf.float32)
    return x

