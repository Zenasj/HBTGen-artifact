# tf.random.uniform((B, 3), dtype=tf.float32) ← Input shape (batch, features=3), inferred from example Dense layer input shapes

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create a Dense layer with 3 units matching the example usage
        self.dense = tf.keras.layers.Dense(3, activation="relu")
        
        # After building the layer, separate bias weight from trainable_weights
        # We'll override the trainable_weights and non_trainable_weights lists to simulate
        # making only bias non-trainable while weight matrix (kernel) trainable.
        # This workaround uses internal lists rather than layer.trainable, since
        # keras doesn't support setting individual variable's trainable attribute.
        
        # Build the layer by passing a dummy input
        self.dense.build(input_shape=(None, 3))
        
        # kernel weights variable
        self.kernel = self.dense.kernel
        # bias variable
        self.bias = self.dense.bias
        
        # By default both are trainable
        # To "freeze" only bias: 
        # - Remove bias from trainable_weights list (variables to be updated)
        # - Add bias to non_trainable_weights list
        
        # Remove bias from trainable weights if present
        if self.bias in self.dense.trainable_weights:
            self.dense.trainable_weights.remove(self.bias)
        
        # Add bias to non-trainable weights if not present
        if self.bias not in self.dense.non_trainable_weights:
            self.dense.non_trainable_weights.append(self.bias)

    def call(self, inputs):
        # Forward pass through the dense layer with kernel trainable, bias fixed.
        # Since bias is in non_trainable_weights, optimizer will ignore it.
        return self.dense(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor matching model's expected input shape: (batch_size, 3)
    # batch size arbitrarily chosen as 4 here.
    return tf.random.uniform((4, 3), dtype=tf.float32)

