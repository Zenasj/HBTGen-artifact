# tf.random.uniform((B, 50), dtype=tf.float32) ← input shape inferred from code (batch dimension variable)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple binary classification model architecture as per example:
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Create an instance of MyModel, compile it with Adam optimizer,
    # matching the original example: learning_rate=0.0005
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model

def GetInput():
    # Return a random tensor with appropriate shape: (batch size, 50 features)
    # Use fixed batch size 32 as a typical example
    return tf.random.uniform((32, 50), dtype=tf.float32)

# Additional helper functions not asked but relevant for the discussed TPU optimizer weights load logic.
# TensorFlow's TPU strategy requires some care when loading optimizer states,
# this snippet shows how user might save and reload weights per the issue discussion.

def save_optimizer_weights(model, optimizer, path):
    """
    Save optimizer weights to disk.
    Note: Cast first weight (usually iteration count) to int32 as done in TPU example.
    """
    # Apply zero grads to initialize optimizer variables
    grad_vars = model.trainable_weights
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    optimizer.apply_gradients(zip(zero_grads, grad_vars))
    opt_weights = optimizer.get_weights()
    # Cast first element to int32 to avoid issues on TPU loading
    opt_weights = [np.int32(opt_weights[0])] + opt_weights[1:]
    np.save(path, opt_weights)

def load_optimizer_weights_on_tpu(model, optimizer, path, strategy):
    """
    Load optimizer weights inside TPU strategy scope,
    applying zero grads first, then setting optimizer weights.

    Copied logic and ordering as recommended in TPU discussion:
    - Load weights outside tf.function
    - Apply zero grads inside tf.function run by strategy
    - Set optimizer weights inside strategy scope and eager context
    """
    opt_weights = np.load(path, allow_pickle=True)
    # Convert numpy arrays to tensors
    opt_weights_tf = [tf.constant(w) for w in opt_weights]

    grad_vars = model.trainable_weights
    zero_grads = [tf.zeros_like(w) for w in grad_vars]

    @tf.function
    def apply_zero_grads():
        optimizer.apply_gradients(zip(zero_grads, grad_vars))

    # Run zero grads application distributed on TPU
    strategy.run(apply_zero_grads)

    # Set the optimizer weights eagerly in strategy scope
    with strategy.scope():
        optimizer.set_weights(opt_weights_tf)

