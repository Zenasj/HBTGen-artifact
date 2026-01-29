# tf.random.uniform((B, 100, 100, 3), dtype=tf.float32)  ← Based on the Keras Conv2D example input_shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # First Conv stack similar to Keras Sequential example
        self.conv_stack1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='valid'),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='valid'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25)
        ])

        # Second Conv stack
        self.conv_stack2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='valid'),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='valid'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25)
        ])

        # Dense layers after flattening
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_final = tf.keras.layers.Dropout(0.5)
        self.dense_out = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv_stack1(inputs, training=training)
        x = self.conv_stack2(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout_final(x, training=training)
        out = self.dense_out(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()

    # Since there's no pretrained weights given, just return the model as is.
    # Note: Compilation etc. is not done here as it's usually done externally.
    return model

def GetInput():
    # Return a random tensor matching the expected input shape
    # Batch size 1 to keep example simple; can adjust as needed.
    return tf.random.uniform((1, 100, 100, 3), dtype=tf.float32)

