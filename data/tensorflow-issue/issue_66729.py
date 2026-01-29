# tf.random.uniform((1, 7), dtype=tf.float32) ← Inferred input shape from example: batch size 1, sequence length 7

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer with 21 outputs
        self.dense = tf.keras.layers.Dense(21)
    
    def call(self, inputs):
        # inputs expected shape: (1, 7) as per corrected shape in the colab & comments
        # The original code tried to reshape a Keras Input tensor using tf.reshape, which caused errors.
        # Here, we assume inputs come in shape (1, 7) already.
        # Simply pass through dense layer.
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel.
    # This model matches the working pattern advised: input tensor shape (1, 7)
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected shape (1,7) and dtype float32
    # This corresponds to batch_size=1, feature_dim=7.
    return tf.random.uniform((1, 7), dtype=tf.float32)

