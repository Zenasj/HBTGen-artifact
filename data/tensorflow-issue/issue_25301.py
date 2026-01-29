# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST images reshaped to (28,28,1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        reg_weight = 0.00001
        regularizers = tf.keras.regularizers
        
        # Equivalent model from the issue, sequential layers converted to functional style within subclass
        self.reshape = tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28))
        self.conv1 = tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation='relu',
            kernel_regularizer=regularizers.l1(reg_weight), use_bias=False)  # use_bias=False for BN compatibility
        self.conv2 = tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu',
            kernel_regularizer=regularizers.l1(reg_weight), use_bias=False)  # use_bias=False to follow with BN
        self.bn = tf.keras.layers.BatchNormalization(fused=True)  # fused batch norm as per discussion
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            128, activation='relu',
            kernel_regularizer=regularizers.l1(reg_weight))
        self.dense2 = tf.keras.layers.Dense(
            10, activation='softmax',
            kernel_regularizer=regularizers.l1(reg_weight))

    def call(self, inputs, training=False):
        # Model forward pass replicating the original issue's model
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x, training=training)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random tensor input matching MNIST images reshaped to (28, 28, 1)
    # Batch size = 1 for simplicity
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

