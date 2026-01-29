# tf.random.uniform((1, 64, 64, 3), dtype=tf.float32)  ← Assuming input is a batch of 64x64 RGB images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialization to replicate the TestModel conv-layer and following layers
        self.kernel_initializer = tf.keras.initializers.HeNormal()
        self.kernel_regularizer = tf.keras.regularizers.l2
        self.l2_reg = 1e-4
        
        self.b_0_0_conv = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1),
                                                 strides=(2, 2),
                                                 padding="same",
                                                 kernel_initializer=self.kernel_initializer,
                                                 kernel_regularizer=self.kernel_regularizer(self.l2_reg),
                                                 name="in_conv")

        self.b_0_0_bn = tf.keras.layers.BatchNormalization(axis=-1, name="in_bn")
        self.b_0_0_relu = tf.keras.layers.ReLU(name="in_relu")
        self.b_0_0_mp = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="in_mp")

    def call(self, x, training=False):
        # Forward pass identical to the reported model
        x = self.b_0_0_conv(x)
        x = self.b_0_0_bn(x, training=training)
        x = self.b_0_0_relu(x)
        x = self.b_0_0_mp(x)
        org = x
        
        # Return two copies as list, matching original behavior
        # We also provide a comparison output showing elementwise equality for debugging
        # This is useful since the issue involves these copies being unexpectedly different on GPU delegate.
        equal_tensors = tf.reduce_all(tf.equal(org, x))
        diff = tf.abs(org - x)
        
        # Output a dict with the two tensors and comparison info
        return {"tensor1": org, "tensor2": x, "equal": equal_tensors, "abs_diff": diff}

def my_model_function():
    # Return an instance of MyModel with fresh weights, suitable for standalone usage
    return MyModel()

def GetInput():
    # Generate a random input tensor compatible with MyModel's conv2d expected input.
    # Based on conv2d with strides=2, kernel=1, input needs to have at least spatial size > 4 for non-trivial output.
    # Assuming input images are RGB with shape (batch, height, width, channels)
    # Guessing input size as 64x64x3 and batch size 1 as a reasonable default.
    return tf.random.uniform((1, 64, 64, 3), dtype=tf.float32)

