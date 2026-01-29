# tf.random.uniform((B, 4), dtype=tf.float32)  ← Input shape is (batch_size, 4)

import tensorflow as tf

K = tf.keras.backend

def Loss(y_true, y_pred):
    # Custom loss function based on the original issue:
    # mean of maximum between |y0/(|y|+0.1)| and |y/(|y0|+0.1)|
    return K.mean(K.maximum(K.abs(y_true / (K.abs(y_pred) + 0.1)), K.abs(y_pred / (K.abs(y_true) + 0.1))))

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer with 1 output unit and bias, matching the example
        self.dense = tf.keras.layers.Dense(1, use_bias=True)

    def call(self, inputs, training=False):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model with the custom loss and Adam optimizer, matching example usage
    model.compile(loss=Loss, optimizer=tf.keras.optimizers.Adam())
    return model

def GetInput():
    # Return a random float32 tensor matching input shape `(batch_size, 4)`.
    # Batch size arbitrary, here I choose 8 for demo.
    return tf.random.uniform((8, 4), dtype=tf.float32)

