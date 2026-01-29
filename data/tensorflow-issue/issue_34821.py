# tf.random.uniform((BATCH_SIZE * NUM_WORKERS, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers as in build_and_compile_cnn_model
        self.conv2d = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.maxpool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_relu = tf.keras.layers.Dense(64, activation='relu')
        self.dense_softmax = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv2d(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense_relu(x)
        return self.dense_softmax(x)


def my_model_function():
    # Create an instance of MyModel and compile it as per original code
    model = MyModel()
    # Compile with sparse categorical crossentropy loss and SGD optimizer with lr=0.001
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Based on the issue, the batch size is GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS (assumed NUM_WORKERS=2)
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    global_batch_size = BATCH_SIZE * NUM_WORKERS
    # Input shape = (batch_size, 28, 28, 1), float32 scaled 0..1
    # Use uniform distribution to simulate scaled pixel data
    return tf.random.uniform(
        shape=(global_batch_size, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)

