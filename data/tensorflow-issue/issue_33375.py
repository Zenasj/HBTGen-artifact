# tf.random.uniform((2, 32, 32, 3), dtype=tf.float32) ← inferred input shape from usage (batch 2, image 32x32, 3 channels)

import tensorflow as tf
import tensorflow_addons as tfa

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple convolutional model matching the original code
        self.conv = tf.keras.layers.Conv2D(3, 3, padding="same")
        self.loss_obj = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(self):
        # Random angle scalar between 0 and pi (3.14)
        angles_rad = tf.random.uniform((), 0, 3.14)
        # Input images batch (2, 32, 32, 3) uniform random float32
        images = tf.random.uniform((2, 32, 32, 3), dtype=tf.float32)

        with tf.GradientTape() as tape:
            features = self.conv(images)
            rot_features = tfa.image.rotate(features, angles_rad, interpolation='NEAREST')
            loss = self.loss_obj(images, rot_features)

        variables = self.conv.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    @tf.function
    def call(self, inputs, training=False):
        # Inference call - just conv layer
        return self.conv(inputs)

def my_model_function():
    # Return an instance of MyModel with initialized layers and optimizer
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape expected by MyModel
    # Batch 2, height 32, width 32, channels 3, dtype float32
    return tf.random.uniform((2, 32, 32, 3), dtype=tf.float32)

