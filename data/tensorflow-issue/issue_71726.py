# tf.random.uniform((32, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Data augmentation layers
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
        ])

        # Main CNN architecture
        self.rescaling = tf.keras.layers.Rescaling(1./255)
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Note: softmax units will be set dynamically in call() based on class count

        # We do not set final Dense in init because class count is unknown here.

    def build(self, input_shape):
        # Infer number of classes from input shape if input is a tuple (X,label) or context
        # In practice, output classes must be provided by caller or fixed during initialization.
        # For this example, assume 10 classes as placeholder.
        num_classes = 10
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

        super().build(input_shape)

    def call(self, inputs, training=False):
        # Inputs shape expected: (B, 224, 224, 3)
        x = self.data_augmentation(inputs, training=training)
        x = self.rescaling(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.output_layer(x)


def my_model_function():
    """
    Returns an instance of MyModel.
    Note:
      - The output classification layer currently assumes 10 classes.
      - For real use, modify MyModel to accept number of classes as constructor argument
        or to dynamically build the last layer based on dataset.
    """
    model = MyModel()
    # Build model with expected input shape to create weights before usage
    model.build((None, 224, 224, 3))
    return model


def GetInput():
    """
    Returns a random input tensor compatible with MyModel.
    Shape: (32, 224, 224, 3)
    dtype: tf.float32
    """
    return tf.random.uniform((32, 224, 224, 3), dtype=tf.float32)

