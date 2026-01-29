# tf.random.uniform((8, 100, 100, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the CNN model layers as described in the example from the issue
        self.conv_1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='conv_1', input_shape=(100, 100, 1))
        self.max_pool_1 = tf.keras.layers.MaxPooling2D(2, name='max_pool_1')
        self.dropout_1 = tf.keras.layers.Dropout(0.2, name='dropout_1')

        self.conv_2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', name='conv_2')
        self.max_pool_2 = tf.keras.layers.MaxPooling2D(2, name='max_pool_2')
        self.dropout_2 = tf.keras.layers.Dropout(0.2, name='dropout_2')

        self.conv_3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', name='conv_3')
        self.max_pool_3 = tf.keras.layers.MaxPooling2D(2, name='max_pool_3')
        self.dropout_3 = tf.keras.layers.Dropout(0.2, name='dropout_3')

        self.flatten = tf.keras.layers.Flatten(name='flatten')

        self.dense_1 = tf.keras.layers.Dense(128, activation='relu', name='dense_1')
        self.dense_2 = tf.keras.layers.Dense(64, activation='relu', name='dense_2')
        self.dense_3 = tf.keras.layers.Dense(10, activation='softmax', name='dense_3')

    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.max_pool_1(x)
        x = self.dropout_1(x, training=training)

        x = self.conv_2(x)
        x = self.max_pool_2(x)
        x = self.dropout_2(x, training=training)

        x = self.conv_3(x)
        x = self.max_pool_3(x)
        x = self.dropout_3(x, training=training)

        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x

def my_model_function():
    # Return a new instance of the model
    model = MyModel()
    # Build the model by providing a dummy input shape
    # This ensures variables are created and weights initialized
    model.build(input_shape=(None, 100, 100, 1))
    return model

def GetInput():
    # Generate a batch of 8 random images of size 100x100 with 1 channel (grayscale)
    # Matches the input shape expected by MyModel
    return tf.random.uniform((8, 100, 100, 1), dtype=tf.float32)

