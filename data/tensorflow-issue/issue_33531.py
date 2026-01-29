# tf.random.uniform((12, 372, 558, 3), dtype=tf.float32) ← inferred from DenseNet121 example inputs (batch_size=12, height=372, width=558, channels=3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using DenseNet121 as the backbone model, no pretrained weights,
        # input shape (372, 558, 3), and 10 classes as per provided snippet.
        self.densenet = tf.keras.applications.DenseNet121(
            weights=None,
            input_shape=(372, 558, 3),
            classes=10)
        # Build explicitly with batch size 12 for static shape
        self.densenet.build((12, 372, 558, 3))

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # Forward pass through DenseNet121
        # inputs shape: (12,372,558,3)
        return self.densenet(inputs, training=training)

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Returns random tensor matching input shape for DenseNet121, batch size=12,
    # height=372, width=558, channels=3, dtype float32.
    return tf.random.uniform((12, 372, 558, 3), dtype=tf.float32)

