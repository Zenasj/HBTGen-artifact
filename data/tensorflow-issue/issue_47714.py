# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Mimic the model architecture from the issue's build_model()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(49, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        # Forward pass same as the sequential model
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    model = MyModel()
    # Compile with same settings as in the issue
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Mimic the preprocessed MNIST input: shape (B, 28, 28, 1),
    # normalized roughly in [-1, +1]
    batch_size = 8
    # Use tf.random.uniform to generate float32 in [-1, 1]
    x = tf.random.uniform((batch_size, 28, 28, 1), minval=-1.0, maxval=1.0, dtype=tf.float32)
    return x

