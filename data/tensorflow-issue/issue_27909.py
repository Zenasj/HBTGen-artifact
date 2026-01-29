# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← input shape inferred from MNIST data preprocessing, default float32/cast_dtype

import tensorflow as tf

NUM_CLASSES = 10
IMAGE_ROW, IMAGE_COLS = 28, 28
BATCH_SIZE = 32

class MyModel(tf.keras.Model):
    def __init__(self, input_shape=(28, 28, 1), 
                 kernel_size=(3, 3), dropout_rate=0.0, l2_regularizer=0.1):
        super(MyModel, self).__init__()
        # Define layers following the functional model from the issue
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, activation='relu')
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)
        # Final dense with L2 regularization as described
        self.predictions = tf.keras.layers.Dense(
            NUM_CLASSES,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(l=l2_regularizer))
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.predictions(x)

def my_model_function():
    # Returns an instance of MyModel with default parameters matching MNIST input shape
    # (28, 28, 1), no dropout by default, L2 regularizer of 0.1 as in the original
    return MyModel(input_shape=(IMAGE_ROW, IMAGE_COLS, 1), dropout_rate=0.0, l2_regularizer=0.1)

def GetInput():
    # Return random input tensor matching shape and dtype expected by MyModel
    # Batch size can be arbitrary, here set to B=1 for convenience
    B = 1
    # Input dtype should be float32 as inferred from preprocessing (cast_dtype)
    return tf.random.uniform((B, IMAGE_ROW, IMAGE_COLS, 1), dtype=tf.float32)

