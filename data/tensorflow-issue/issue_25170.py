# tf.random.uniform((B, 50, 100, 1), dtype=tf.float32)
import tensorflow as tf

DATA_SHAPE = (50, 100, 1)  # Input shape: single channel 2D "image" tensor

class SimpleResNet(tf.keras.Model):
    def __init__(self, n_filters=10):
        super().__init__()
        # First Conv2D block
        self.conv1 = tf.keras.layers.Conv2D(n_filters, kernel_size=(3, 3), padding='same', use_bias=False)
        self.relu1 = tf.keras.layers.ReLU()
        self.bn1 = tf.keras.layers.BatchNormalization()

        # Second Conv2D block
        self.conv2 = tf.keras.layers.Conv2D(n_filters, kernel_size=(3, 3), padding='same', use_bias=False)
        self.relu2 = tf.keras.layers.ReLU()
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Lambda layer for global average pooling per channel (mean over height and width)
        self.global_avg_pool = tf.keras.layers.Lambda(lambda z: tf.math.reduce_mean(z, axis=[1, 2]))
        # Dense output with 2 classes (logits)
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        x = inputs
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.bn1(res, training=training)
        res = self.conv2(res)
        res = self.relu2(res)
        res = self.bn2(res, training=training)

        x = x + res  # Residual connection

        x = self.global_avg_pool(x)
        x = self.dense(x)
        return x

class SimpleResNetNoBN(tf.keras.Model):
    """
    Simplified version of SimpleResNet with BatchNormalization layers removed.
    This is to mimic the workaround suggested by the issue comments where removing BN 
    resolved TFLite discrepancy issues.
    """
    def __init__(self, n_filters=10):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(n_filters, kernel_size=(3,3), padding='same', use_bias=False)
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(n_filters, kernel_size=(3,3), padding='same', use_bias=False)
        self.relu2 = tf.keras.layers.ReLU()

        self.global_avg_pool = tf.keras.layers.Lambda(lambda z: tf.math.reduce_mean(z, axis=[1, 2]))
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        x = inputs
        res = self.conv1(x)
        res = self.relu1(res)
        # No BN here
        res = self.conv2(res)
        res = self.relu2(res)
        # No BN here

        x = x + res

        x = self.global_avg_pool(x)
        x = self.dense(x)
        return x

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We include both original model with BN and model without BN as submodules
        self.model_with_bn = SimpleResNet()
        self.model_without_bn = SimpleResNetNoBN()

    def call(self, inputs, training=False):
        # Run both models on the input
        output_with_bn = self.model_with_bn(inputs, training=training)
        output_without_bn = self.model_without_bn(inputs, training=training)

        # Compute difference between outputs to highlight BN effect discrepancy
        diff = tf.abs(output_with_bn - output_without_bn)

        # Return output tuple with three components:
        #  1) Output from model with BN (represents original trained model)
        #  2) Output from model without BN (represents simplified model to check BN effect)
        #  3) Difference between outputs (to analyze discrepancy)
        return output_with_bn, output_without_bn, diff

def my_model_function():
    # Create an instance of MyModel
    return MyModel()

def GetInput():
    # Generate random input tensor with shape (1, 50, 100, 1) matching DATA_SHAPE used in the models
    # Use tf.random.uniform with dtype float32 (common input dtype for image-like tensors)
    input_tensor = tf.random.uniform(shape=(1, *DATA_SHAPE), minval=0, maxval=1, dtype=tf.float32)
    return input_tensor

