# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← MNIST images, grayscale, 28x28

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.features = tf.keras.layers.Dense(128, activation='relu', name='features')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.features(x)
        x = self.dropout2(x, training=training)
        outputs = self.classifier(x)
        return outputs

def my_model_function():
    model = MyModel()
    # Compile as per original example
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor shaped as a MNIST batch of 128 images with 1 channel
    # Use float32 normalized data in range [0,1]
    batch_size = 128
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32, minval=0.0, maxval=1.0)

