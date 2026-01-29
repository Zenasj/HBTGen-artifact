# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the original sequential layers as submodules
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1))
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.logits = tf.keras.layers.Dense(10)  # output layer, logits
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        
        x = self.logits(x)  # logits without softmax due to from_logits=True in loss
        return x

def my_model_function():
    """
    Returns an instance of MyModel.
    This model needs to be compiled after creation with:
        model.compile(optimizer='adam',
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['sparse_categorical_accuracy'])
    Note: Use 'sparse_categorical_accuracy' explicitly to avoid accuracy mismatch bug present in TF 2.3.0.
    """
    return MyModel()

def GetInput():
    # Generate a random batch of 32 grayscale images 28x28 with 1 channel, dtype float32 in [0,1]
    B = 32
    return tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)

