# tf.random.uniform((128, 28, 28, 1), dtype=tf.float32) ← inferred input shape from MNIST data format 'channels_last'

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        super().__init__()
        # Define the Conv2D, pooling, dropout, flatten, dense layers mirroring the Sequential model from issue
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel initialized with MNIST input shape and 10 classes
    return MyModel(input_shape=(28,28,1), num_classes=10)

def GetInput():
    # Generate a batch of random MNIST-like inputs with shape (128, 28, 28, 1), dtype float32, values in [0,1]
    # Batch size 128 matches example batch_size from issue code
    return tf.random.uniform((128, 28, 28, 1), dtype=tf.float32)

