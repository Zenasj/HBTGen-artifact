import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf 
import numpy as np 

class SSIMMetric(tf.keras.metrics.Metric):
    """
    Custom SSIM metric for image denoising.

    This class calculates the average Structural Similarity Index Measure (SSIM)
    across all batches during training or evaluation.

    Args:
      name (str, optional): A name for the metric. Defaults to 'ssim'.
      max_val (float, optional): The dynamic range of the images (usually the maximum pixel value).
          Defaults to 255.0 for images in the 0-255 range.
      **kwargs (optional): Additional keyword arguments for the base Metric class.
    """

    def __init__(self, name='ssim', max_val=255.0, **kwargs):
        super(SSIMMetric, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.ssim = self.add_variable(shape=(), name='ssim', initializer='zeros')
        self.counter = self.add_variable(shape=(), name='counter', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the internal state of the metric with a batch of data.

        Args:
            y_true (tf.Tensor): The ground truth image tensor.
            y_pred (tf.Tensor): The predicted image tensor.
            sample_weight (tf.Tensor, optional): Sample weights (not used in this implementation).
        """
        ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=self.max_val))
        self.ssim.assign((ssim + self.ssim * self.counter) / (self.counter + 1.))
        self.counter.assign_add(1.)
        
    def result(self):
        """
        Calculates and returns the current average SSIM value.

        Returns:
            tf.Tensor: The average SSIM across all batches.
        """
        return self.ssim 

    def reset_states(self):
        """
        Resets the internal state of the metric to zero.
        """
        self.ssim.assign(0.0)
        self.counter.assign(0.0)


class PSNRMetric(tf.keras.metrics.Metric):
    """
    Custom PSNR metric for image denoising.

    This class calculates the average Peak Signal-to-Noise Ratio (PSNR)
    across all batches during training or evaluation.

    Args:
      name (str, optional): A name for the metric. Defaults to 'psnr'.
      max_val (float, optional): The dynamic range of the images (usually the maximum pixel value).
          Defaults to 255.0 for images in the 0-255 range.
      **kwargs (optional): Additional keyword arguments for the base Metric class.
    """
    def __init__(self, name='psnr', max_val=255.0, **kwargs):
        super(PSNRMetric, self).__init__(name=name, **kwargs)
        self.max_val = max_val
        self.psnr = self.add_variable(shape=(), name='psnr', initializer='zeros')
        self.counter = self.add_variable(shape=(), name='counter', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the internal state of the metric with a batch of data.

        Args:
            y_true (tf.Tensor): The ground truth image tensor.
            y_pred (tf.Tensor): The predicted image tensor.
            sample_weight (tf.Tensor, optional): Sample weights (not used in this implementation).
        """
        psnr = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=self.max_val))
        self.psnr.assign((psnr + self.counter * self.psnr) / (self.counter + 1.))
        self.counter.assign_add(1.)

    def result(self):
        """
        Calculates and returns the current average PSNR value.

        Returns:
            tf.Tensor: The average PSNR across all batches.
        """
        return self.psnr 

    def reset_states(self):
        """
        Resets the internal state of the metric to zero.
        """
        self.psnr.assign(0.0)
        self.counter.assign(0.0)
        
        

(y_train, _), (y_test, _)   = tf.keras.datasets.cifar10.load_data()

y_train, y_test = y_train/255.0, y_test / 255.0 

x_train = y_train + 0.2 * np.random.random((y_train.shape))
x_test  = y_test  + 0.2 * np.random.random((y_test.shape))

input_layer = tf.keras.layers.Input((32, 32, 3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), 
                           activation=tf.nn.leaky_relu, padding='same')(input_layer)
x = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), 
                           activation='sigmoid', padding='same')(x)

model = tf.keras.models.Model(input_layer, x)

model.compile(loss='mse', metrics=[PSNRMetric(max_val=1.), SSIMMetric(max_val=1.)], optimizer='Adam')

results = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

val_psnr = np.array([np.array(metric_val) for metric_val in results.history['val_psnr']])
psnr     = np.array([np.array(metric_val) for metric_val in results.history['psnr']])
print(f'training psnr:\n{psnr}\nval_psnr:\n{val_psnr}')