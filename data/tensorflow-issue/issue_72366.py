# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← Input shape is CIFAR-10 images batch with height=32, width=32, channels=3

import tensorflow as tf

class SSIMMetric(tf.keras.metrics.Metric):
    """
    Custom SSIM metric for image denoising.

    Calculates the average Structural Similarity Index Measure (SSIM)
    across all batches during training or evaluation.

    Args:
      name (str, optional): A name for the metric. Defaults to 'ssim'.
      max_val (float, optional): The dynamic range of the images (usually max pixel value).
          Defaults to 1.0 assuming normalized images in [0,1].
      **kwargs (optional): Additional keyword arguments for base Metric class.
    """
    def __init__(self, name='ssim', max_val=1.0, **kwargs):
        super(SSIMMetric, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        # Use the modern add_weight API, as add_variable is deprecated.
        self.ssim = self.add_weight(shape=(), name='ssim', initializer='zeros')
        self.counter = self.add_weight(shape=(), name='counter', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute SSIM for batch and update running average
        ssim_batch = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=self.max_val))
        new_ssim = (ssim_batch + self.ssim * self.counter) / (self.counter + 1)
        self.ssim.assign(new_ssim)
        self.counter.assign_add(1)

    def result(self):
        return self.ssim

    def reset_states(self):
        self.ssim.assign(0.0)
        self.counter.assign(0.0)


class PSNRMetric(tf.keras.metrics.Metric):
    """
    Custom PSNR metric for image denoising.

    Calculates the average Peak Signal-to-Noise Ratio (PSNR)
    across all batches during training or evaluation.

    Args:
      name (str, optional): A name for the metric. Defaults to 'psnr'.
      max_val (float, optional): The dynamic range of images.
          Defaults to 1.0 assuming normalized images in [0,1].
      **kwargs (optional): Additional keyword arguments for base Metric class.
    """
    def __init__(self, name='psnr', max_val=1.0, **kwargs):
        super(PSNRMetric, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.psnr = self.add_weight(shape=(), name='psnr', initializer='zeros')
        self.counter = self.add_weight(shape=(), name='counter', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnr_batch = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=self.max_val))
        new_psnr = (psnr_batch + self.psnr * self.counter) / (self.counter + 1)
        self.psnr.assign(new_psnr)
        self.counter.assign_add(1)

    def result(self):
        return self.psnr

    def reset_states(self):
        self.psnr.assign(0.0)
        self.counter.assign(0.0)


class MyModel(tf.keras.Model):
    """
    A simple convolutional autoencoder for denoising CIFAR-10 images.

    The architecture from the reported issue:
    - Input shape: (32, 32, 3)
    - Conv2D with 16 filters, 3x3 kernel, leaky ReLU activation, padding same
    - Conv2D with 3 filters, 3x3 kernel, sigmoid activation, padding same

    This model denoises noisy inputs to reconstruct original images.
    """

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), 
                                            activation=tf.nn.leaky_relu, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), 
                                            activation='sigmoid', padding='same')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor matching CIFAR-10 normalized images shape: (B, 32, 32, 3)
    # Use batch size 8 as a reasonable default
    batch_size = 8
    # Values in [0, 1] consistent with normalized images for metric max_val=1.0
    return tf.random.uniform((batch_size, 32, 32, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

