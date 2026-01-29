# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST example (channels_last)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # CNN architecture roughly matching the MNIST example shown in the issue
        
        # Conv2D layers
        self.conv1 = tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            dilation_rate=(1, 1),
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu',
            input_shape=(28, 28, 1))
        
        self.conv2 = tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            dilation_rate=(1, 1),
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation='relu')
        
        # Pooling and dropout layers
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        
        # Flatten and dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            128,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros')
        
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        
        self.dense2 = tf.keras.layers.Dense(
            10,  # Number of classes for MNIST
            activation='softmax',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel, as in the tf.keras example.
    # No weights loading since none provided; user can compile and train after.
    return MyModel()

def GetInput():
    # Return a random input tensor matching the MNIST input shape used in model: (batch, 28, 28, 1)
    # Batch size arbitrarily chosen as 16 for demonstration.
    batch_size = 16
    height, width, channels = 28, 28, 1
    # Random floats in [0, 1), matching preprocessing in the example.
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

