# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assumed input shape for typical image data like MNIST (28x28 grayscale)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple normalization layer simulating the _map_fn example: scale pixel values to [0,1]
        # Since the original issue involves None values in dataset mapping,
        # here we model a typical normalization and include a metadata output as None,
        # but since TensorFlow cannot output None tensors, we instead output a tf.zeros with shape ().
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)
    
    def call(self, inputs, training=False):
        # inputs shape assumed: (B, H, W, C)
        image_norm = self.normalize(inputs)
        label_placeholder = tf.zeros(shape=(tf.shape(inputs)[0],), dtype=tf.int32)
        # We cannot output None, so output a tensor to simulate 'metadata' placeholder
        metadata_placeholder = tf.zeros(shape=(tf.shape(inputs)[0],), dtype=tf.float32)
        # Return a dictionary simulating DatasetOutputs as NamedTuple
        return {"image": image_norm, "label": label_placeholder, "metadata": metadata_placeholder}

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor mimicking an MNIST-like batch image input,
    # batch size 4, 28x28 grayscale images (C=1)
    return tf.random.uniform((4, 28, 28, 1), minval=0, maxval=255, dtype=tf.float32)

