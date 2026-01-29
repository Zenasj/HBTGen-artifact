# tf.random.uniform((B, 784), dtype=tf.float32)  ← input shape inferred from keras.Input(shape=(784,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Setup mixed precision policy to match the example in the issue (mixed_float16)
        # Note: Normally this would be done externally, but included here for completeness.
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        self.dense = tf.keras.layers.Dense(10)
        # Activation layer with output dtype float32 (as in example)
        self.activation = tf.keras.layers.Activation('softmax', dtype='float32')
        
    def call(self, inputs, training=False):
        x = self.dense(inputs)
        outputs = self.activation(x)
        return outputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Batch size chosen as 32 for example; could be any batch size.
    return tf.random.uniform((32, 784), dtype=tf.float32)

