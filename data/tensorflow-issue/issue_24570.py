# tf.random.uniform((6, 251, 388, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Build a CNN model similar to the one described in the issue
        # Assumptions:
        # - Input shape: (251, 388, 1) grayscale images as per ImageDataGenerator target size and batch size 6
        # - 2 conv layers with 64 filters, kernel size 3x3, relu activation, padding same
        # - 1 max pooling layer with pool size 2x2, stride 2x2, valid padding
        # - Flatten and Dense output layer with softmax activation over 3 classes
        self.conv1 = tf.keras.layers.Conv2D(
            64, (3,3), strides=(1,1), activation='relu', padding='same', input_shape=(251,388,1)
        )
        self.conv2 = tf.keras.layers.Conv2D(
            64, (3,3), activation='relu', padding='same'
        )
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()

    # Compile the model (following issue details)
    # Optimizer: Adam with learning rate 0.001
    # Loss: categorical_crossentropy (multi-class classification)
    # Metrics: accuracy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor matching the expected input shape and dtype to directly feed into MyModel
    # Batch size 6 assumed from original code's ImageDataGenerator batch size
    # Input shape is (6, 251, 388, 1), dtype float32 (images)
    return tf.random.uniform(shape=(6, 251, 388, 1), minval=0, maxval=1, dtype=tf.float32)

