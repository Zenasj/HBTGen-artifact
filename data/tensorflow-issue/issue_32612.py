# tf.random.uniform((B, 32), dtype=tf.float32) ← Input shape is inferred from example Input layer shape=(32,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple example model: one Dense layer mapping 32 features to 32 outputs
        self.dense = tf.keras.layers.Dense(32)

        # Custom loss instance embedded as submodule
        self.my_loss = MyCustomLoss()

    def call(self, inputs, training=None):
        return self.dense(inputs)
    
    def compute_loss(self, y_true, y_pred):
        # Provide a wrapper for loss computation like Keras expects
        return self.my_loss(y_true, y_pred)


class MyCustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        # No additional state or config needed in this minimal example

    def call(self, y_true, y_pred):
        # Return constant loss as in the example (always 1)
        return tf.constant(1.0)

    def get_config(self):
        # Required to enable saving/loading the custom loss with config
        config = super().get_config()
        return config

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor with shape (batch_size, 32)
    # Batch size chosen as 4 (arbitrary) for example usage
    return tf.random.uniform((4, 32), dtype=tf.float32)

