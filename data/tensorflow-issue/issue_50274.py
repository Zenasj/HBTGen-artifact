# tf.random.uniform((1, 96, 112, 3), dtype=tf.float32) ← inferred input shape based on the ResNet50 snippet with input_shape=[96,112,3]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load base ResNet50 model without top layer and pretrained ImageNet weights
        # input_shape matches the reported input shape used in the issue (96,112,3)
        self.base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(96, 112, 3)
        )
        self.flatten = tf.keras.layers.Flatten()
        # Bottleneck dense layer with 128 units and bias enabled, named 'Bottleneck'
        self.bottleneck = tf.keras.layers.Dense(128, use_bias=True, name='Bottleneck')

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.flatten(x)
        x = self.bottleneck(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input with shape [1, 96, 112, 3] matching the expected input shape
    # Using uniform random data here as a placeholder since the original input was a zero matrix but random data 
    # better tests full graph execution and avoids constant folding differences.
    return tf.random.uniform((1, 96, 112, 3), dtype=tf.float32)

