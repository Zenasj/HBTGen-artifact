# tf.random.normal((32, 32, 32, 8), dtype=...) ← Input shape inferred from benchmark and model: batch=32?, height=32, width=32, channels=8 (initial conv outputs 8 channels)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Per issue discussion and performance problem:
        # Use float16 mixed precision generally (per policy),
        # but force DepthwiseConv2D layer to float32 to avoid slow/unstable fp16 backprop filter kernel.
        # The original example input shape is (32, 32, 3) from CIFAR10, but benchmark uses (32, 32, 32, 8).
        # We'll reconstruct the toy model roughly from the issue:
        self.conv2d = tf.keras.layers.Conv2D(
            8, 3, padding="same", activation="relu",
            input_shape=(32, 32, 3),
            dtype="mixed_float16")  # standard AMP layer

        # Enforce float32 dtype explicitly on DepthwiseConv2D to mitigate slowdown as recommended
        self.depthwise_conv2d = tf.keras.layers.DepthwiseConv2D(
            3, depth_multiplier=8, padding="same", activation="relu",
            dtype="float32")  # workaround per issue

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(dtype="float32")  # safe to use float32 here
        self.dense = tf.keras.layers.Dense(10, dtype="float32")  # final classification layer
        self.softmax = tf.keras.layers.Activation("softmax", dtype="float32")

    def call(self, inputs, training=None):
        # Follow the original flow:
        x = self.conv2d(inputs)
        # Cast input to depthwise_conv2d to float32 since its input is float16 by default via conv2d output.
        # But conv2d outputs mixed_float16 (usually float16), so we cast to float32 explicitly here to match depthwise layer dtype
        x = tf.cast(x, tf.float32)
        x = self.depthwise_conv2d(x)
        x = self.global_avg_pool(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Set mixed precision policy for the model globally
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    model = MyModel()
    # Compile with optimizer and loss as in the original example
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random tensor input compatible with MyModel input:
    # Assuming batch size 32 (per benchmarks), height=32, width=32, channels=3 (CIFAR10 image)
    # dtype float32 here, since input preprocessing does normalization and float32 cast.
    # The model first conv2d layer expects float16 input per policy, but it's okay to pass float32
    # TensorFlow will cast as necessary.
    batch_size = 32
    height = 32
    width = 32
    channels = 3
    # Using uniform or normal is arbitrary, here normal per benchmark.
    input_tensor = tf.random.normal((batch_size, height, width, channels), dtype=tf.float32)
    return input_tensor

