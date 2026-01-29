# tf.random.uniform((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32)
import tensorflow as tf
import tensorflow_datasets as tfds

IMG_HEIGHT = 200
IMG_WIDTH = 200
BATCH_SIZE = 128

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.bn1 = tf.keras.layers.BatchNormalization(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu')
        self.pool1 = tf.keras.layers.MaxPooling2D(2)
        self.drop1 = tf.keras.layers.Dropout(0.25)

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='elu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2)
        self.drop2 = tf.keras.layers.Dropout(0.25)

        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(64, (5, 5), activation='elu')
        self.pool3 = tf.keras.layers.MaxPooling2D(2)
        self.drop3 = tf.keras.layers.Dropout(0.25)

        self.bn4 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='elu')
        self.pool4 = tf.keras.layers.MaxPooling2D(2)
        self.drop4 = tf.keras.layers.Dropout(0.25)

        self.bn5 = tf.keras.layers.BatchNormalization()
        self.conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='elu')
        # Commented out MaxPooling2D as it was disabled in original
        # self.pool5 = tf.keras.layers.MaxPooling2D(2)
        self.drop5 = tf.keras.layers.Dropout(0.25)

        self.bn6 = tf.keras.layers.BatchNormalization()
        self.conv6 = tf.keras.layers.Conv2D(512, (3, 3), activation='elu')
        self.pool6 = tf.keras.layers.MaxPooling2D(2)
        self.drop6 = tf.keras.layers.Dropout(0.25)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256)
        self.elu = tf.keras.layers.Activation('elu')
        self.drop7 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10)
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=False):
        x = self.bn1(inputs, training=training)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        x = self.bn2(x, training=training)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        x = self.bn3(x, training=training)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        x = self.bn4(x, training=training)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.drop4(x, training=training)

        x = self.bn5(x, training=training)
        x = self.conv5(x)
        # x = self.pool5(x)  # Pooling commented out to match original code
        x = self.drop5(x, training=training)

        x = self.bn6(x, training=training)
        x = self.conv6(x)
        x = self.pool6(x)
        x = self.drop6(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.elu(x)
        x = self.drop7(x, training=training)
        x = self.dense2(x)
        return self.softmax(x)


def my_model_function():
    # Return an instance of MyModel initialized with compiled optimizer and loss
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    return model


@tf.function
def convert(image, label):
    # Convert uint8 image to float32 in [0,1], label to float32 and expand dims as per original code
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.expand_dims(tf.cast(label, tf.float32), 0)
    return image, label


@tf.function
def resize(image, label):
    # Resize image to (IMG_HEIGHT, IMG_WIDTH)
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    return image, label


def GetInput():
    """
    Returns a batch of random input images with shape (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1),
    matching the expected input shape of MyModel.
    """
    # Using uniform random tensor in [0,1], dtype float32
    return tf.random.uniform(
        (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1),
        minval=0.0, maxval=1.0, dtype=tf.float32
    )

