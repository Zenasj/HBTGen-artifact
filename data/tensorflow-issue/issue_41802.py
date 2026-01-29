# tf.random.uniform((20, 150, 150, 3), dtype=tf.float32) ← batch_size=20, image size 150x150 with 3 channels (RGB)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Sequential-like CNN architecture for binary classification (cats vs dogs)
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D(2, 2)
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2, 2)
        self.conv3 = tf.keras.layers.Conv2D(128, (3,3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(2, 2)
        self.conv4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(2, 2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output (sigmoid)

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
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    model = MyModel()
    # Compile model with RMSprop optimizer as per original
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return random batch of images with the expected input shape (20 images, 150x150 RGB)
    # Using float32 dtype normalized similar to rescale=1./255 (between 0 and 1)
    return tf.random.uniform(shape=(20, 150, 150, 3), minval=0, maxval=1, dtype=tf.float32)

