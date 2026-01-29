# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← assumed CIFAR-10 input shape, batch size unspecified

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assumptions based on the issue:
        # - Input shape corresponds to CIFAR-10 images: (32, 32, 3)
        # - Conv2DTranspose first layer requires input_shape to avoid initialization error with Identity kernel_initializer
        # - Use filters=11, kernel_size=19, strides=19, padding='valid', activation='relu', kernel_initializer='Identity'
        #   (as in the suggested fix in the issue)
        # - Flatten and Dense(num_classes=100) follow, no activation on final Dense (logits)
        # Note: kernel_initializer='Identity' requires a 2D kernel shape which usually isn't possible for Conv2DTranspose.
        # The 'Identity' initializer is only valid for 2D square matrices, 
        # so assuming the fix is to specify input_shape to allow this usage in the first Conv2DTranspose layer.
        
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters=11,
            kernel_size=19,
            strides=19,
            padding='valid',
            activation='relu',
            kernel_initializer='Identity',
            input_shape=(32, 32, 3)  # CIFAR-10 size input shape
        )
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.dense = tf.keras.layers.Dense(100)  # 100 classes like CIFAR-100

    def call(self, inputs, training=False):
        x = self.conv_transpose(inputs)
        x = self.flatten(x)
        logits = self.dense(x)
        return logits


def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching expected input: batch size arbitrary (e.g., 28),
    # height=32, width=32, channels=3 for CIFAR images, dtype float32, normalized [0,1]
    B = 28
    H = 32
    W = 32
    C = 3
    return tf.random.uniform((B, H, W, C), minval=0, maxval=1, dtype=tf.float32)

