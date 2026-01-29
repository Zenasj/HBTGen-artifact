# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the model like the "make_model" in the provided snippet
        self.conv = tf.keras.layers.Conv2D(
            32, 3, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.02),
            input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.batchnorm(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel; weights are uninitialized here
    return MyModel()

def GetInput():
    # Return a random tensor simulating a batch of grayscale 28x28 images
    # Match batch size from the code snippet: 64
    BATCH_SIZE = 64
    return tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)

