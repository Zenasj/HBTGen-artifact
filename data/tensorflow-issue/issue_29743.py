# tf.random.uniform((batch_size, 64, 23), dtype=tf.float32) ← inferred input shape from code snippet and shapes used

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assuming input shape is (64, 23) as per the original code
        self.reshape = tf.keras.layers.Reshape((64, 23, 1))
        self.conv1 = tf.keras.layers.Conv2D(8, (30, 23), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D((4, 5), strides=(4, 5))
        self.conv2 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        # Note: original model had Dense with 1 unit and softmax, which is unusual for binary classification.
        # Typically sigmoid is used for binary outputs. We'll keep softmax as per original but note.
        self.dense = tf.keras.layers.Dense(1, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.flatten(x)
        out = self.dense(x)
        return out

def my_model_function():
    # Return an instance of MyModel as defined above
    return MyModel()

def GetInput():
    # Generate a random input tensor that matches expected input
    # The batch size is chosen arbitrarily (e.g., 10) to align with example usage
    batch_size = 10
    return tf.random.uniform((batch_size, 64, 23), dtype=tf.float32)

