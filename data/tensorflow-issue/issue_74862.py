# tf.random.uniform((B, H, W, C), dtype=tf.float32)  # Assumed input shape (batch, H, W, C) inferred from model input shape in lenet(inp.shape[1:])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the sub-model as per lenet function in the issue
        
        # Activation layers in sequence
        self.act_relu = tf.keras.layers.Activation(activation="relu")
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=8, strides=3, padding='same')
        self.act_sigmoid = tf.keras.layers.Activation(activation='sigmoid')
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=5, strides=1, padding='same')
        self.act_softplus = tf.keras.layers.Activation(activation='softplus')
        self.maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=7, strides=8, padding='same', data_format='channels_last')
        
        # Flatten with channels_first data_format (note: input likely channels_last, so we preserve this part as is)
        self.flatten1 = tf.keras.layers.Flatten(data_format='channels_first')
        
        # Dense layers with activation='exponential'
        # From issue code, first Dense has:
        # units=2, activation='exponential', use_bias=False, kernel_initializer='ones'
        self.dense_exp1 = tf.keras.layers.Dense(
            units=2, activation='exponential', use_bias=False, kernel_initializer='ones')
        
        # Second Dense layer has:
        # units=7, activation='exponential', use_bias=True, kernel_initializer='glorot_uniform'
        self.dense_exp2 = tf.keras.layers.Dense(
            units=7, activation='exponential', use_bias=True, kernel_initializer='glorot_uniform')
        
        # Flatten again with channels_first
        self.flatten2 = tf.keras.layers.Flatten(data_format='channels_first')
        
        # Final Dense layer with units=10, no activation specified (linear)
        self.dense_final = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=None):
        x = self.act_relu(inputs)
        x = self.maxpool1(x)
        x = self.act_sigmoid(x)
        x = self.maxpool2(x)
        x = self.act_softplus(x)
        x = self.maxpool3(x)
        
        x = self.flatten1(x)
        x = self.dense_exp1(x)
        x = self.dense_exp2(x)
        x = self.flatten2(x)
        x = self.dense_final(x)
        return x


def my_model_function():
    # Instantiate MyModel and return it
    return MyModel()


def GetInput():
    # To generate a valid input tensor:
    # From the original code, the input shape is inferred from inp.shape[1:] in lenet.
    # We don't have the exact input dimensions in the issue, but based on typical lenet usage and the 
    # usage of maxpool with large kernel sizes and strides, let's assume a 4D input tensor with shape:
    # (batch_size, height, width, channels) = (1, 56, 56, 3)
    #
    # This is an assumption for demonstration since the exact shape is not given.
    #
    # Use dtype float32 for compatibility.
    
    B = 1
    H = 56
    W = 56
    C = 3
    
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)

