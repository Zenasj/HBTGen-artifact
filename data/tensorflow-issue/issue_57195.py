# tf.random.uniform((B, 300, 300, 3), dtype=tf.float32) ← Input shape inferred from Encoder input shape in the issue

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Lambda
import math
import cv2
import numpy as np

# The original issue uses a log-polar transform on feature maps.
# This requires converting tensors to numpy arrays, but inside tf.function this causes errors.
# Here, we implement a tf-compatible workaround using tf.numpy_function to safely wrap OpenCV calls.

def log_polar_sampling_np(feature_maps_np):
    # feature_maps_np: numpy array of shape (H,W,C)
    # Convert each channel to log-polar using OpenCV warpPolar.
    h, w, c = feature_maps_np.shape
    log_polar = np.zeros_like(feature_maps_np)
    center = (w / 2, h / 2)
    radius = int(w * 0.9)
    M = w / math.log(radius)
    flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
    dim = (100, 100)
    for i in range(c):
        image = feature_maps_np[:, :, i]
        log_polar_img = cv2.warpPolar(image, dim, center, M, flags)
        # Resize output to match input spatial dims (H,W)
        log_polar_img_resized = cv2.resize(log_polar_img, (w, h), interpolation=cv2.INTER_CUBIC)
        log_polar[:, :, i] = log_polar_img_resized
    return log_polar.astype(np.float32)

def log_polar_sampling(feature_maps):
    # feature_maps: tf.Tensor with shape (batch, H, W, C)
    # We want to apply log-polar transform per batch sample
    def map_fn(img):
        # img shape: (H,W,C)
        return tf.numpy_function(log_polar_sampling_np, [img], tf.float32)
    # Apply per example in batch: shape transforms as batch dims preserved
    out = tf.map_fn(map_fn, feature_maps, fn_output_signature=tf.float32)
    return out

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Encoder layers defined to match the architecture in the issue
        self.conv1_1 = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")
        self.conv1_2 = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")
        self.pool1 = MaxPool2D(pool_size=(2,2), strides=2)

        self.conv2_1 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")
        self.conv2_2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")
        self.pool2 = MaxPool2D(pool_size=(2,2), strides=2)

        self.conv3_1 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")
        self.conv3_2 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")
        self.conv3_3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")
        self.pool3 = MaxPool2D(pool_size=(2,2), strides=2)

        self.conv4_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
        self.conv4_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
        self.conv4_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
        self.pool4 = MaxPool2D(pool_size=(2,2), strides=2)

        self.conv5_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
        self.conv5_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")
        self.conv5_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")

        # Following original model logic, add log-polar transform as Lambda layer here
        # replaced by method call in forward pass for clarity and control

        self.fc6 = Conv2D(filters=1024, kernel_size=(3,3), padding="same")
        self.fc7 = Conv2D(filters=1024, kernel_size=(1,1), padding="same")

        self.conv8_1 = Conv2D(filters=256, kernel_size=(1,1), padding="same")
        self.conv8_2 = Conv2D(filters=256, kernel_size=(3,3), padding="same", strides=2)

        self.conv9_1 = Conv2D(filters=128, kernel_size=(1,1), padding="same")
        self.conv9_2 = Conv2D(filters=256, kernel_size=(3,3), padding="same", strides=2)

        self.conv10_1 = Conv2D(filters=128, kernel_size=(1,1), padding="same")
        self.conv10_2 = Conv2D(filters=256, kernel_size=(3,3), padding="same", strides=1)

        self.conv11_1 = Conv2D(filters=128, kernel_size=(1,1), padding="same")
        self.conv11_2 = Conv2D(filters=256, kernel_size=(3,3), padding="same", strides=1)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        # Apply log-polar transform to conv5_3 output
        x = log_polar_sampling(x)

        x = self.fc6(x)
        x = self.fc7(x)

        x = self.conv8_1(x)
        x = self.conv8_2(x)

        x = self.conv9_1(x)
        x = self.conv9_2(x)

        x = self.conv10_1(x)
        x = self.conv10_2(x)

        x = self.conv11_1(x)
        x = self.conv11_2(x)

        return x

def my_model_function():
    # Returns an instance of MyModel, ready for inference/training
    model = MyModel()
    # Model weights uninitialized; typical usage is to call model.build or run dummy input through it
    # Alternatively, model can be compiled and trained afterward
    return model

def GetInput():
    # Return a random float32 tensor matching the input shape expected by MyModel: (batch, 300, 300, 3)
    # Batch dimension assumed 1 for simplicity
    return tf.random.uniform((1, 300, 300, 3), dtype=tf.float32)

