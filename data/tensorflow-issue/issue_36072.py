# tf.random.uniform((B, 160, 160, 3), dtype=tf.float32) ← inferred input shape from issue's MobileNetV2 example with (160,160,3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the issue code, base model is MobileNetV2 feature extractor with input_shape=(160,160,3)
        # followed by GlobalAveragePooling2D, BatchNormalization, Dense(2000 classes), softmax activation.
        # We build same architecture here.

        self.base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            input_shape=(160, 160, 3),
            weights=None,  # weights=None to match example without pretrained weights
            pooling=None,
            classes=2000
        )
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(2000)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.global_pool(x)
        x = self.batch_norm(x, training=training)
        x = self.dense(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Return an instance of MyModel, compiled similarly to issue example's model.compile
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def GetInput():
    # Generate a random input tensor with shape matching model input,
    # dtype float32, range [0,1], like issue's preprocess scaling x/255
    # Batch size can be arbitrary; use batch size 8 as a reasonable default
    batch_size = 8
    input_shape = (batch_size, 160, 160, 3)
    return tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)

