# tf.random.uniform((B, 32, 32, 3), dtype=tf.float64) ← Based on CIFAR-10 dataset input shape and dtype set to float64 as in original snippet

import tensorflow as tf
from tensorflow.keras import layers, models

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Build Model1 components explicitly in __init__
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3), name='model1_conv2d_1')
        self.pool1 = layers.MaxPooling2D((2, 2), name='model1_maxpool_1')

        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', name='model1_conv2d_2')
        self.pool2 = layers.MaxPooling2D((2, 2), name='model1_maxpool_2')

        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu', name='model1_conv2d_3')

        # Model2 components:
        # From the text, Model2 first reshapes input to (32, 32, 1),
        # but since the original input is (32, 32, 3), this does not match directly.
        # Assuming Model2 expects grayscale images. For simplicity,
        # convert input to grayscale in call method or adjust pipeline.
        self.reshape = layers.Reshape((32,32,1), name='model2_reshape')
        self.flatten = layers.Flatten(name='model2_flatten')
        self.dense1 = layers.Dense(64, activation='relu', name='model2_dense_1')
        self.dense2 = layers.Dense(10, name='model2_dense_2')

    def call(self, inputs):
        """
        Run the ensemble model:
        1. Pass input through Model1 convolutional layers.
        2. Convert Model1 output appropriately to feed into Model2, here flattening Model1 conv features,
           or alternatively, preprocess original inputs separately for Model2.
        3. Pass preprocessed input to Model2 layers.
        4. For demonstration, here Model2 operates on Model1 output flattened to vector.
           Alternatively, Model2 could independently process inputs.
        """

        # Model1 forward pass
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Flatten Model1 output as input to Model2
        x_flat = self.flatten(x)

        # For Model2, the original design does reshape input to (32,32,1) then flattens, but this is unclear in ensemble
        # So here Model2 processes Model1 features, for ensemble chaining.
        # Instead of reshaping Model1 output (which is incompatible), we treat reshaping as no-op or skip.
        # The original reshape might be to reshape grayscale images, but here we pass x_flat directly.

        # Model2 dense layers using Model1 features:
        x2 = self.dense1(x_flat)
        output = self.dense2(x2)

        return output


def my_model_function():
    # Return an instance of MyModel, set dtype to float64 as per original example
    model = MyModel()
    return model

def GetInput():
    # Return a random tensor input matching (batch_size, 32, 32, 3) with dtype float64
    # Batch size: default to 1 to allow model call without errors
    return tf.random.uniform((1, 32, 32, 3), dtype=tf.float64)

