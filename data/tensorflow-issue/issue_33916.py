# tf.random.uniform((10, 224, 224, 3), dtype=tf.float32) ← inferred input shape based on batch size and Conv2D input_shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the original Sequential model described:
        # 7 Conv2D layers with filters=1, kernel sizes 3 or 2,
        # strides mostly 2, final activation sigmoid.
        self.conv_layers = [
            tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, activation=None, input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, activation=None),
            tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, activation=None),
            tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, activation=None),
            tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, activation=None),
            tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, activation=None),
            tf.keras.layers.Conv2D(filters=1, kernel_size=2, strides=1, activation='sigmoid'),
        ]
        
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model similarly as original code:
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy')
    return model

def GetInput():
    # Return a random tensor input matching model input
    # batch size=10, height=224, width=224, channels=3, dtype float32
    return tf.random.uniform(shape=(10, 224, 224, 3), dtype=tf.float32)

