# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build LeNet-5 style architecture exactly as described in the issue
        self.conv1 = tf.keras.layers.Conv2D(
            6, kernel_size=(3, 3), activation='relu', name="conv1")
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")
        self.conv2 = tf.keras.layers.Conv2D(
            16, kernel_size=(3, 3), activation='relu', name="conv2")
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.dense1 = tf.keras.layers.Dense(120, activation='relu', name="dense1")
        self.dense2 = tf.keras.layers.Dense(84, activation='relu', name="dense2")
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax', name="output")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.output_layer(x)
        return output

def my_model_function():
    """
    Returns:
        MyModel: an instance of the LeNet-5 style model.
        
    Note: 
    - Weights are expected to be loaded separately by user if available.
    - Compiled with Adam optimizer and categorical cross-entropy loss.
    """
    model = MyModel()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])
    return model

def GetInput():
    """
    Returns:
        tf.Tensor: A random tensor simulating a batch of grayscale 28x28 images,
                   shape (batch_size, 28, 28, 1), dtype float32.
    """
    batch_size = 1  # Default batch size of 1; can be increased as needed
    return tf.random.uniform(shape=(batch_size, 28, 28, 1), dtype=tf.float32)

