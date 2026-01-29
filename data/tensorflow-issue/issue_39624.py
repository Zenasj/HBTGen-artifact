# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← input shape inference, assume a batch of images similar to 'binary_alpha_digits' dataset; let's assume images are 8x8 grayscale (as the dataset 'binary_alpha_digits' has 8x8 grayscale images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # According to the original issue, the model has:
        # - Dense layer with 16 units, kernel_initializer=he_normal (fixed issue in newer TF versions)
        # - Dense output layer with units equal to the number of classes in dataset 'binary_alpha_digits' (36 classes: 10 digits + 26 letters)
        # Since we cannot rely on tensorflow_datasets in this isolated code, we hardcode number of classes =36
        # Input shape is implicit, images flattened before dense layers
        
        # He normal initializer can be used directly in TF 2.20.0 without dtype argument errors
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(16, kernel_initializer=tf.keras.initializers.HeNormal())
        self.layer2 = tf.keras.layers.Dense(36)  # 36 classes in `binary_alpha_digits`
    
    def call(self, inputs, training=None, **kwargs):
        x = self.flatten(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def my_model_function():
    # Return an instance of MyModel as defined above
    return MyModel()

def GetInput():
    # Return a batch of example inputs with shape matching the expected input to MyModel:
    # According to tfds.binary_alpha_digits dataset: images are 8x8 grayscale, batch size 8 in example code
    # Make dtype float32 as in the original mapping (tf.cast(..., tf.float32))
    batch_size = 8
    height = 8
    width = 8
    channels = 1
    # Return a tensor of random floats simulating a batch of 8 grayscale images 8x8
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

