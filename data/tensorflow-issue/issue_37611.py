# tf.random.uniform((16, 150, 150, 3), dtype=tf.float32) ← Inferred input shape from original model input_shape=(150, 150, 3), batch_size=16

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the CNN model as per provided Sequential model architecture
        self.conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3))
        self.pool1 = tf.keras.layers.MaxPooling2D(2,2)

        self.conv2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2,2)

        self.conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(2,2)

        self.conv4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(2,2)

        self.flatten = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.output_layer = tf.keras.layers.Dense(5, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of MyModel, no pretrained weights to load (based on issue context)
    return MyModel()

def GetInput():
    # Return random tensor matching expected input shape and dtype
    # batch_size inferred as 16 in original code (batch_size=16)
    # dtype float32, pixel values normalized between 0 and 1
    batch_size = 16
    height = 150
    width = 150
    channels = 3
    # Create uniform random tensor (simulate normalized image batch)
    return tf.random.uniform((batch_size, height, width, channels), minval=0, maxval=1, dtype=tf.float32)

