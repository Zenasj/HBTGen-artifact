# tf.random.uniform((B, 192, 192, 3), dtype=tf.float32)  ← Input shape inferred from MobileNetV2 input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Pretrained MobileNetV2 without top layers
        self.mobile_net = tf.keras.applications.MobileNetV2(
            input_shape=(192, 192, 3),
            include_top=False,
            weights='imagenet'
        )
        # Following the example structure: GlobalAveragePooling2D + Dense(5)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')
        self.dense = tf.keras.layers.Dense(5, name='dense')

    def call(self, inputs, training=False):
        # Pass input through MobileNetV2 backbone
        x = self.mobile_net(inputs, training=training)
        # Then through global average pooling
        x = self.global_avg_pool(x)
        # Then to dense layer for final output
        x = self.dense(x)
        return x


def my_model_function():
    # Return a new instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches MobileNetV2's expected input shape
    # Use batch size of 1 as a default
    return tf.random.uniform(shape=(1, 192, 192, 3), dtype=tf.float32)

