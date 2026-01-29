# tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)  ← Assumed standard input shape for convolutional MNIST-like example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model reflects a quantization-aware training friendly convnet inspired by discussed examples
        # BatchNormalization layers use fused=False as recommended in issue comments to avoid TOCO quantization errors
        self.conv1 = tf.keras.layers.Conv2D(
            16, kernel_size=3, activation='relu', padding='same', use_bias=False, input_shape=(28, 28, 1))
        self.bn1 = tf.keras.layers.BatchNormalization(fused=False)
        
        self.conv2 = tf.keras.layers.Conv2D(
            32, kernel_size=3, strides=2, activation='relu', padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(fused=False)
        
        self.conv3 = tf.keras.layers.Conv2D(
            64, kernel_size=3, strides=2, activation='relu', padding='same', use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(fused=False)
        
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=7)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)  # training argument important for batchnorm
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel.
    # Quantization aware training would typically apply further graph rewrites
    # after building the model, e.g., tf.contrib.quantize.create_training_graph().
    # That logic is not included here (TF1 contrib quantize deprecated).
    # This is a KO-compatible base model structure, using fused=False on BatchNorm layers.
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Using batch size 1, height=28, width=28, channels=1 grayscale image,
    # dtype float32, values in [0, 1] typical for image preprocessing.
    return tf.random.uniform(shape=(1, 28, 28, 1), dtype=tf.float32)

