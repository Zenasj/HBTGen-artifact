# tf.random.uniform((B, 500, 500, 1), dtype=tf.float32) ← Input shape inferred from resize_image_500 output and Conv2D layers

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(500,500,1))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.conv4 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1024, activation='relu')

        # Note: The original code had no final classification layer 
        # with softmax or logits. Typically for classification you'd add:
        # self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')
        # But since label quantification and number of classes is unknown here,
        # we keep it as in original.

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
    # Return an instance of MyModel
    model = MyModel()
    # Compile with reasonable placeholders for a classification problem.
    # The original used CategoricalCrossentropy (likely for one-hot labels),
    # but metrics.Accuracy does not work well with sparse labels directly.
    # We replicate the setup as close as possible.
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy()]
    )
    return model

def GetInput():
    # Return a random tensor input matching the expected input to MyModel
    # Batch size arbitrary, e.g. 8
    batch_size = 8
    # Single channel grayscale images 500x500
    return tf.random.uniform((batch_size, 500, 500, 1), dtype=tf.float32)

