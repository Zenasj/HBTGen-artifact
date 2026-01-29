# tf.random.uniform((B, 5), dtype=tf.float32) ← Input shape inferred as (batch_size, 5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, inp_dim=5, outp_dim=5):
        super().__init__()
        # Simple dense layer with tanh activation as in the reported model
        self.dense = tf.keras.layers.Dense(outp_dim, activation='tanh', name='simple_layer')
        self.inp_dim = inp_dim
        
    def call(self, inputs):
        # inputs is a tuple of (anchor, positive)
        anchor, positive = inputs  # both shape (batch_size, inp_dim)
        outp_anchor = self.dense(anchor)     # (batch_size, outp_dim)
        outp_positive = self.dense(positive) # (batch_size, outp_dim)
        
        u = outp_anchor - outp_positive      # difference vector
        
        # The reported issue:
        # when u is zero vector (i.e. anchor == positive after model),
        # norm(u) is zero and gradients become NaN.
        # Workaround is to replace zeros with a small positive value or epsilon,
        # or use some "double-where" trick to avoid zero norm.

        # Implement double-where trick below:
        # Replace values close to zero in u with 1.0 element-wise for norm calculation,
        # This avoids norm being exactly zero and thus NaNs in gradients.
        epsilon = 1e-6
        u_safe = tf.where(tf.abs(u) > 0.001, u, tf.ones_like(u))  # replace near-zero entries with 1
        u_norm = tf.linalg.norm(u_safe, axis=-1)  # shape (batch_size,)
        
        # This outputs the norm (loss value) per batch item,
        # User would typically reduce mean or sum outside this model.
        return u_norm

def my_model_function():
    # Return an instance of MyModel with default dimensions as per original example
    return MyModel(inp_dim=5, outp_dim=5)

def GetInput():
    # Generate a tuple of two inputs (anchor, positive)
    # Both random floats of shape (batch_size, 5)
    batch_size = 10
    inp_dim = 5

    # Generate random uniform inputs in range [0, 1)
    anchor = tf.random.uniform((batch_size, inp_dim), dtype=tf.float32)
    positive = tf.random.uniform((batch_size, inp_dim), dtype=tf.float32)

    # Force positive[0] = anchor[0] to simulate the similar batch input
    positive = tf.concat([anchor[0:1], positive[1:]], axis=0)
    return (anchor, positive)

