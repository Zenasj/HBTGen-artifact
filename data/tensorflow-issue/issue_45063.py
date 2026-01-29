# tf.random.uniform((1, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST data_shape (28,28,1) from tfds

import tensorflow as tf
import tensorflow_probability as tfp

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct the same convolutional stack as in the issue, since that triggers the problem
        # Use layers similar to the original model for faithful reproduction.
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, use_bias=False)
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.softplus = tf.keras.layers.Activation(tf.nn.softplus)
        self.conv_last = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, use_bias=False)
        self.flatten = tf.keras.layers.Flatten()

        # Recreate latent distribution matching dimension of flattened output gradient
        # The dimension is 28x28x1 = 784 as per MNIST
        self.dimension = 28 * 28 * 1
        self.latent_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self.dimension),
            scale_diag=tf.ones(self.dimension),
        )
        # Optimizer from original code (though not strictly necessary for forward call)
        self.optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def call(self, inputs, training=False):
        # Forward pass of model as defined
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.softplus(x)  # Crucial to trigger the known bug in the issue
        x = self.conv_last(x)
        x = self.flatten(x)
        return x

    @tf.function  # The training step containing nested GradientTapes
    def train_step(self, data):
        # data assumed batched input images of shape (batch, 28, 28, 1), tf.float32 normalized [0, 1]
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)

            with tf.GradientTape() as c_tape:
                c_tape.watch(data)

                with tf.GradientTape() as a_tape:
                    a_tape.watch(data)
                    # Forward pass
                    b = self.call(data, training=True)
                # First derivative of output w.r.t. input
                a = a_tape.gradient(b, data)  # shape same as input
                a_flat = tf.reshape(a, (-1, self.dimension))

            # Jacobian of a w.r.t. input; shape batch x dimension x dimension
            c = c_tape.batch_jacobian(a, data)
            c = tf.reshape(c, (-1, self.dimension, self.dimension))

            d = self.latent_distribution.log_prob(a_flat)
            _, e = tf.linalg.slogdet(c)
            ff = tf.reduce_mean(d + e)

            loss = -ff  # objective to minimize

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


def my_model_function():
    # Instantiate the model, weights will be randomly initialized (no pretrained weights available)
    return MyModel()

def GetInput():
    # Return a single batch of MNIST-like input: batch size 1, 28x28 grayscale, float32 normalized [0,1]
    # Using tf.random.uniform with shape (1, 28, 28, 1).
    return tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)

