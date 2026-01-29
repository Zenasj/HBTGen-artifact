# tf.random.uniform((B, 100), dtype=tf.float32) ← Inferred input shape from the issue: batch size B and feature dimension 100

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers matching the original example
        self.dense1 = tf.keras.layers.Dense(10)
        self.dense2 = tf.keras.layers.Dense(15)
        self.dense3 = tf.keras.layers.Dense(5)
        self.dense4 = tf.keras.layers.Dense(5)
        
        # L2 regularizer instance
        self.l2_regularizer = tf.keras.regularizers.l2(1e-4)
        
        # After building the model, we add regularization losses correctly
        # We will add these losses capturing the variable in closure properly,
        # to avoid the common lambda-in-loop late binding problem
        
        # Collect the layers that should be regularized
        self._layers_to_regularize = [self.dense1, self.dense2, self.dense3, self.dense4]
        
        # Add losses in a way that closures capture layers correctly
        for layer in self._layers_to_regularize:
            self.add_loss(self._make_l2_loss_fn(layer))
    
    def _make_l2_loss_fn(self, layer):
        # We define a function returning the l2 loss on layer.kernel
        # Capturing layer as a fixed argument in the closure avoids the
        # common lambda-in-loop trap
        def l2_loss():
            return self.l2_regularizer(layer.kernel)
        return l2_loss

    @tf.function(jit_compile=True)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

def my_model_function():
    # Instantiate and return the MyModel instance
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape:
    # batch size arbitrary (e.g., 4), feature size 100 as per example
    return tf.random.uniform((4, 100), dtype=tf.float32)

