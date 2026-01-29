# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)  # Assumed NHWC input shape as typical image classifier input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assumptions:
        # The original model is a CNN classifier on input (1, 224, 224, 3)
        # Since the original PyTorch model processes inputs of shape (1,3,224,224) (NCHW),
        # we will assume a similar architecture but adapted to TF using typical Conv2D layers.
        # The channel numbers and kernel sizes are inferred from the PyTorch params:
        # params.channels = [6,12,24]
        # params.kernel_widths = [3,4,2]
        # params.pooling_widths = [3,4,2]
        # We will use strides and pool sizes accordingly.
        # For simplicity dropout rates as in PyTorch params: conv_dropout_rate=0.2, fc_dropout_rate=0.2
        # Fully connected layers hidden sizes: [128, 256, 128]
        # Output classes: 10 (non-regressor)
        
        self.conv1 = tf.keras.layers.Conv2D(6, kernel_size=(1,3), padding='valid', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(1,3))
        self.drop1 = tf.keras.layers.Dropout(0.2)
        
        self.conv2 = tf.keras.layers.Conv2D(12, kernel_size=(1,4), padding='valid', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(1,4))
        self.drop2 = tf.keras.layers.Dropout(0.2)
        
        self.conv3 = tf.keras.layers.Conv2D(24, kernel_size=(1,2), padding='valid', activation='relu')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(1,2))
        self.drop3 = tf.keras.layers.Dropout(0.2)
        
        # Flatten after conv layers
        self.flatten = tf.keras.layers.Flatten()
        
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.drop_fc1 = tf.keras.layers.Dropout(0.2)
        
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.drop_fc2 = tf.keras.layers.Dropout(0.2)
        
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.drop_fc3 = tf.keras.layers.Dropout(0.2)
        
        self.output_layer = tf.keras.layers.Dense(10)  # 10 classes
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        
    def call(self, inputs, training=False):
        # Inputs expected shape: (batch_size, 224, 224, 3)
        # To imitate PyTorch's (N, C, H, W) with C=3,
        # we assume input channels last format as usual for TensorFlow
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.drop1(x, training=training)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.drop3(x, training=training)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.drop_fc1(x, training=training)
        
        x = self.fc2(x)
        x = self.drop_fc2(x, training=training)
        
        x = self.fc3(x)
        x = self.drop_fc3(x, training=training)
        
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random tensor matching the expected input shape (batch=1, height=224, width=224, channels=3)
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

