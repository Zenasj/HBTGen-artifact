# tf.random.uniform((B, 198, 198, 3), dtype=tf.float32) ← Input shape corresponds to (height=198, width=198, 3 channels RGB)

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Activation, Dropout,
    Lambda, Dense, Flatten, Input
)
from tensorflow.keras import Model

class MyModel(tf.keras.Model):
    """
    Multi-output CNN for predicting age, race, and gender from input images.
    This is a fused reimplementation based on the UtkMultiOutputModel class
    from the original issue.

    Outputs a dictionary with keys:
      - 'age_output': Regression (linear activation output scalar)
      - 'race_output': Multi-class classification (softmax over races)
      - 'gender_output': Binary classification (sigmoid for 2 classes)

    The network has 3 branches sharing the same base conv layers.
    """

    def __init__(self, width=198, height=198, num_races=5, num_genders=2):
        super().__init__()
        self.input_shape_ = (height, width, 3)
        self.num_races = num_races
        self.num_genders = num_genders

        # Define shared convolutional base layers for default hidden layers method
        self.conv1 = Conv2D(16, (3,3), padding="same")
        self.act1 = Activation("relu")
        self.bn1 = BatchNormalization(axis=-1)
        self.pool1 = MaxPooling2D(pool_size=(3,3))
        self.drop1 = Dropout(0.25)

        self.conv2 = Conv2D(32, (3,3), padding="same")
        self.act2 = Activation("relu")
        self.bn2 = BatchNormalization(axis=-1)
        self.pool2 = MaxPooling2D(pool_size=(2,2))
        self.drop2 = Dropout(0.25)

        self.conv3 = Conv2D(32, (3,3), padding="same")
        self.act3 = Activation("relu")
        self.bn3 = BatchNormalization(axis=-1)
        self.pool3 = MaxPooling2D(pool_size=(2,2))
        self.drop3 = Dropout(0.25)

        # Age branch Dense layers
        self.age_flatten = Flatten()
        self.age_dense1 = Dense(128)
        self.age_act1 = Activation("relu")
        self.age_bn1 = BatchNormalization()
        self.age_drop = Dropout(0.5)
        self.age_output_layer = Dense(1, name="age_output")
        self.age_activation = Activation("linear")

        # Race branch Dense layers
        self.race_flatten = Flatten()
        self.race_dense1 = Dense(128)
        self.race_act1 = Activation("relu")
        self.race_bn1 = BatchNormalization()
        self.race_drop = Dropout(0.5)
        self.race_output_layer = Dense(self.num_races, name="race_output")
        self.race_activation = Activation("softmax")

        # Gender branch Dense layers
        self.gender_rgb2gray = Lambda(lambda c: tf.image.rgb_to_grayscale(c))
        self.gender_flatten = Flatten()
        self.gender_dense1 = Dense(128)
        self.gender_act1 = Activation("relu")
        self.gender_bn1 = BatchNormalization()
        self.gender_drop = Dropout(0.5)
        self.gender_output_layer = Dense(self.num_genders, name="gender_output")
        self.gender_activation = Activation("sigmoid")

    def make_default_hidden_layers(self, x):
        # Conv2D -> Activation -> BN -> MaxPool -> Dropout * 3 blocks
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        return x

    def call(self, inputs, training=False):
        # Age branch
        age_x = self.make_default_hidden_layers(inputs)
        age_x = self.age_flatten(age_x)
        age_x = self.age_dense1(age_x)
        age_x = self.age_act1(age_x)
        age_x = self.age_bn1(age_x, training=training)
        age_x = self.age_drop(age_x, training=training)
        age_x = self.age_output_layer(age_x)
        age_x = self.age_activation(age_x)

        # Race branch
        race_x = self.make_default_hidden_layers(inputs)
        race_x = self.race_flatten(race_x)
        race_x = self.race_dense1(race_x)
        race_x = self.race_act1(race_x)
        race_x = self.race_bn1(race_x, training=training)
        race_x = self.race_drop(race_x, training=training)
        race_x = self.race_output_layer(race_x)
        race_x = self.race_activation(race_x)

        # Gender branch: grayscale input
        gender_x = self.gender_rgb2gray(inputs)
        # The issue code dropped grayscale and reused inputs by mistake;
        # here we apply make_default_hidden_layers to gender_x (1 channel grayscale)
        gender_x = self.make_default_hidden_layers(gender_x)
        gender_x = self.gender_flatten(gender_x)
        gender_x = self.gender_dense1(gender_x)
        gender_x = self.gender_act1(gender_x)
        gender_x = self.gender_bn1(gender_x, training=training)
        gender_x = self.gender_drop(gender_x, training=training)
        gender_x = self.gender_output_layer(gender_x)
        gender_x = self.gender_activation(gender_x)

        return {
            'age_output': age_x,
            'race_output': race_x,
            'gender_output': gender_x
        }


def my_model_function():
    # Return an instance of MyModel configured with default input size 198x198 and race classes=5
    return MyModel(width=198, height=198, num_races=5, num_genders=2)


def GetInput():
    # Create a random tensor simulating a batch of images with correct input shape
    batch_size = 32  # Default batch size similar to example
    input_tensor = tf.random.uniform(shape=(batch_size, 198, 198, 3), dtype=tf.float32)
    return input_tensor

