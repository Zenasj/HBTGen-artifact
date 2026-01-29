# tf.random.uniform((B, 125), dtype=tf.float32) ← input shape inferred from code: (batch_size, 125)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the model from the issue:
        # Sequential:
        #   Dense(9, activation='relu', input_dim=125)
        #   Dense(31, activation='softmax')
        # Use functional API style here inside subclassing
        
        self.dense1 = tf.keras.layers.Dense(9, activation='relu', input_shape=(125,))
        self.dense2 = tf.keras.layers.Dense(31, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return a fresh instance of MyModel, compiled here as the original used adam + categorical_crossentropy
    model = MyModel()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a random input tensor compatible with the model input
    # The original input was shape=(batch_size, 125)
    # We choose batch size 16 as a reasonable default
    batch_size = 16
    return tf.random.uniform((batch_size, 125), dtype=tf.float32)

