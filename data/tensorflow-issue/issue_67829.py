# tf.random.uniform((1, 28, 28, 1), dtype=tf.float32) ← Based on input shape [1, 1, 28, 28] transposed to NHWC (1, 28, 28, 1)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Layers correspond to the original model's architecture described

        # Conv2DTranspose layer with 6 filters, kernel 5x5, stride 1, bias enabled
        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=6, kernel_size=(5, 5), strides=(1, 1), padding="valid",
            use_bias=True, name="conv1_mutated")

        # MaxPooling2D layers with pool size and strides 2x2
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="pool1")
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="pool2")

        # Conv2D layer with 16 filters, kernel size 6x8, stride 1, bias enabled
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(6, 8), strides=(1, 1), padding="valid", use_bias=True, name="conv2_mutated")

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten(name="flatten")

        # Dense layers for fully connected part
        self.fc1 = tf.keras.layers.Dense(units=120, use_bias=True, name="linear1")
        self.relu3 = tf.keras.layers.ThresholdedReLU(theta=0.1, name="relu3_mutated")
        self.fc2 = tf.keras.layers.Dense(units=84, use_bias=True, name="linear2_mutated")
        # Activation after fc2 is erf
        # Last Dense layers
        self.fc3 = tf.keras.layers.Dense(units=10, use_bias=True, name="linear3_mutated")
        self.flatten_tail = tf.keras.layers.Flatten(name="tail_flatten")
        self.tail_fc = tf.keras.layers.Dense(units=10, use_bias=True, name="tail_fc")

    def call(self, inputs):
        # inputs shape expected NHWC format

        # First convolutonal transpose layer + relu
        x = self.conv1(inputs)
        x = tf.nn.relu(x)

        # Zero padding with ((0,0),(0,0)) means no padding, effectively a no-op, so omitted

        # Max Pooling layer 1
        x = self.pool1(x)

        # Zero padding no-op omitted

        # Conv2 layer followed by softsign activation
        x = self.conv2(x)
        x = tf.math.softsign(x)

        # Zero padding no-op omitted

        # Max Pooling layer 2
        x = self.pool2(x)

        # The original code transposes maxpool2 output depending on rank;
        # For 4D tensor NHWC, perm = (0,3,1,2) effectively changing axis:
        # Original shape (batch, h, w, c) -> (batch, c, h, w)
        # This transposition is critical before flattening
        # Confirming shape len(x.shape) = 4, so use perm (0,3,1,2)
        x = tf.transpose(x, perm=[0,3,1,2])

        # Flatten before FC
        x = self.flatten(x)

        # FC layers with nonlinearities as described
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = tf.math.erf(x)
        x = self.fc3(x)

        # Transpose fc3_output similarly by rank; fc3 is 2D (batch, 10) so no transpose needed?
        # According to original code, for 2D output shape (batch,10), len=2 -> perm=(0,1) no change
        # To keep consistent, apply transpose with perm=(0,1), which does nothing
        x = tf.transpose(x, perm=[0,1])

        # Flatten and final dense layer
        x = self.flatten_tail(x)
        x = self.tail_fc(x)

        return x

def my_model_function():
    # Return an instance of MyModel, with no weights initialization here
    model = MyModel()
    # Build the model by calling once to create variables (optional but helpful for TF functions)
    dummy_input = GetInput()
    _ = model(dummy_input)
    return model

def GetInput():
    # Return a random tensor input that matches the expected input: batch =1, height=28, width=28, channels=1
    # dtype float32 to align with original code
    return tf.random.uniform(shape=(1, 28, 28, 1), dtype=tf.float32)

