# tf.random.uniform((1000, 1), dtype=tf.float32) ← Inferred input shape from initial example (m=1000, n=1)

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K

# The issue centers around gradients not flowing when using tf.math.square (also tf.math.pow) 
# directly in a functional model or inside a @tf.function-decorated training step.
# One minimal problematic example was a Conv1D that outputs q, then the model returns -tf.math.square(q).
# The fix or best practice is to wrap such ops in custom layers or Lambda layers to keep gradients intact.
#
# Another larger example has a standard Conv2D-based subclassed model and training loop
# with tf.function and distributed strategy, where saving inside strategy.scope() can raise errors.
#
# To demonstrate a single fused MyModel reflecting these issues,
# we implement the minimal problematic Conv1D + square op model as a submodel,
# plus a Conv2D subclassed model as another submodel.
#
# Then MyModel compares the outputs or returns a tuple of outputs.
# This demonstrates how one might encapsulate these models and their computation with gradient flow intact.
#
# Since the original problem was "No gradients provided for any variable" related to tf.math.square and tf.function,
# here we make sure gradients flow by using layers.Lambda wrapping square operation.
#
# We assume input shape (1000, 1) for conv1d model part.
# For demonstration, conv2d model expects input shape like MNIST (28,28,1).
# We'll just create MyModel assuming a single input tensor matching conv1d input (m=1000, n=1).
#
# If multiple inputs were needed, we could extend GetInput to produce a tuple.
# Here we keep it minimal and coherent.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Submodel 1: minimal Conv1D + negative square, as from initial example.
        self.conv1d = layers.Conv1D(1, 1)
        # Use Lambda layer to wrap square to preserve gradient flow inside model graph.
        self.neg_square = layers.Lambda(lambda x: -tf.math.square(x))
        
        # Submodel 2: a simple Conv2D-based subclassed model for classification
        # adapted from MNIST example but with minimal layers due to input shape constraints.
        self.conv2d = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, x):
        # x assumed shape (B, 1000, 1)
        # Run conv1d submodel
        x1 = self.conv1d(x)              # (B, 1000, 1)
        x1 = self.neg_square(x1)         # (B, 1000, 1)
        
        # For conv2d path, we need to reshape input to 2D images with channel dimension
        # We will mimic an MNIST-like shape by reshaping (B, 28, 28, 1)
        # Here, just to demonstrate, we take first 784 (=28*28) steps from input and reshape;
        # for batches smaller than 784, just tile or truncate.
        # This is an assumption to fuse both models in one.
        
        # Extract or pad input to length 784 for 2D conv:
        B = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        # If seq_len < 784, pad zeros at end; if >784, truncate
        def pad_or_trunc():
            padded = tf.pad(x, [[0,0],[0,784 - seq_len],[0,0]])
            return padded[:, :784, :]
        def trunc():
            return x[:, :784, :]
        x_for_conv2d = tf.cond(seq_len < 784, pad_or_trunc, trunc)
        # Reshape to (B,28,28,1)
        x_for_conv2d = tf.reshape(x_for_conv2d, (B, 28, 28, 1))
        
        x2 = self.conv2d(x_for_conv2d)   # (B, 28, 28, 32)
        x2 = self.flatten(x2)            # (B, 28*28*32)
        x2 = self.dense1(x2)             # (B, 128)
        x2 = self.dense2(x2)             # (B, 10)
        
        # Returning both outputs to reflect fusion of multiple models.
        return x1, x2

def my_model_function():
    # Create and return an instance of MyModel.
    return MyModel()

def GetInput():
    # Return a random input tensor shape (batch=1000, length=1) same as original example
    B = 1000    # batch size from original example
    H = 1       # sequence length is n=1 originally
    # But Conv1D expects 3D shape, (B, length, channels)
    # Original X was shape (m=1000, n=1) - we interpret batch dimension as 1000 samples each with length 1 and one channel
    # So input shape (B=1000, length=1, channels=1)
    input_tensor = tf.random.uniform((B, 1, 1), dtype=tf.float32)
    return input_tensor

