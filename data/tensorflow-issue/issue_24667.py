# tf.random.uniform((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32)
import tensorflow as tf

# Constants inferred from the original issue
BATCH_SIZE = 64
IMG_SIZE = 98
N_CLASSES = 179

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on original model architecture and TensorFlow 2.20 compatibility
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=1, activation='relu', padding='same',
            input_shape=(IMG_SIZE, IMG_SIZE, 1))
        self.avgpool1 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, activation='relu', padding='same')
        self.avgpool2 = tf.keras.layers.AveragePooling2D(pool_size=2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, activation='relu', padding='same')
        self.avgpool3 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1024, activation='relu')
        # The original Dropout rate was 0.8 which is very high. Keep as is for faithfulness
        self.dropout = tf.keras.layers.Dropout(rate=0.8)
        self.output_layer = tf.keras.layers.Dense(units=N_CLASSES, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.avgpool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)


def my_model_function():
    # Create an instance of MyModel
    model = MyModel()
    # Compile with categorical_crossentropy loss, as described in the issue
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def GetInput():
    # Return a random tensor matching input shape expected by MyModel
    # The model input shape is (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1)
    # tf.float32 is typical for image inputs
    return tf.random.uniform(shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32)

