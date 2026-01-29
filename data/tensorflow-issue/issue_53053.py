# tf.random.uniform((batch_size, img_height, img_width, 3), dtype=tf.float32) ← Inferred input: RGB images with shape (img_height, img_width, 3)

import tensorflow as tf
from tensorflow.keras import layers

# A placeholder custom data augmentation layer named RandomColorDistortion was mentioned but not provided.
# To keep the example complete, create a stub layer that just passes inputs through,
# since the original implementation is unknown and causes serialization issues.
class RandomColorDistortion(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        # Parameters are omitted to avoid save/load errors; implement as needed.

    def call(self, inputs):
        # No-op or placeholder color distortion logic
        return inputs

    def get_config(self):
        config = super().get_config()
        # Normally, add parameters here for correct serialization
        return config


class MyModel(tf.keras.Model):
    def __init__(self, img_height=224, img_width=224, num_classes=10,
                 kernel_size=(3,3), padding='same'):
        super().__init__()
        # Save parameters for reference
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.padding = padding

        # Data augmentation pipeline as given in the issue
        self.data_augmentation_rgb = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(0.5),
            layers.RandomZoom(0.5),
            layers.RandomContrast(0.5),
            RandomColorDistortion(name='random_contrast_brightness/none'),
        ])

        # Rescale layer (standardizing inputs to [0,1])
        self.rescale = layers.Rescaling(1./255)

        # Conv blocks as described
        self.conv1 = layers.Conv2D(16, kernel_size, padding=padding, activation='relu', strides=1, data_format='channels_last')
        self.pool1 = layers.MaxPooling2D()
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(32, kernel_size, padding=padding, activation='relu')
        self.pool2 = layers.MaxPooling2D()
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(64, kernel_size, padding=padding, activation='relu')
        self.pool3 = layers.MaxPooling2D()
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(128, kernel_size, padding=padding, activation='relu')
        self.pool4 = layers.MaxPooling2D()
        self.bn4 = layers.BatchNormalization()

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.1)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.1)
        self.dense3 = layers.Dense(64, activation='relu')
        self.dropout3 = layers.Dropout(0.1)
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Apply data augmentation only in training mode
        x = inputs
        if training:
            x = self.data_augmentation_rgb(x)
        x = self.rescale(x)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn3(x, training=training)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.bn4(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        return self.classifier(x)


def my_model_function():
    # Instantiate with default values that can be adjusted as needed
    # E.g. img_height=224, img_width=224, num_classes=10 for a typical dataset
    model = MyModel(img_height=224, img_width=224, num_classes=10)
    # The model can be compiled externally as needed:
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def GetInput():
    # Return a random input tensor with shape (batch_size, img_height, img_width, 3)
    # Using default batch size 4 for demonstration
    batch_size = 4
    img_height = 224
    img_width = 224
    # Random uniform input simulating raw images in [0,1]
    return tf.random.uniform((batch_size, img_height, img_width, 3), dtype=tf.float32)

