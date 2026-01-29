# tf.random.uniform((batch, num_samples), dtype=tf.float32) ← This input shape applies inside the custom categorical function; 
# For MyModel input, inferred input shape is (500, 500) as np.random.rand(500,500) is used as input to Keras model.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the Keras model equivalent to the sequential model in the issue
        # Input shape 500 applies to each input sample, but input data is (500,500) --> meaning batch=500, features=500?
        # Because np.random.rand(500,500) is passed as x, the shape is (batch=500, features=500).
        # So input shape for the Dense model is (500,), and batch size is 500.
        # We'll define input shape for Dense accordingly.
        
        # The keras.Sequential([Input(500,), Dense(500, relu), Dense(act_dim, softmax)]) creates 
        # a model that maps from features=500 -> hidden 500 -> act_dim=8 logits.
        
        self.act_dim = 8
        self.dense1 = tf.keras.layers.Dense(500, activation="relu")
        self.dense2 = tf.keras.layers.Dense(self.act_dim, activation="softmax")
        
    def call(self, inputs, training=False):
        """
        Forward pass:
        - inputs: float32 tensor of shape (batch_size, 500)
        - outputs: categorical samples drawn from the predicted probabilities using a custom categorical sampler
        """
        # Pass through dense layers
        prob = self.dense1(inputs)
        prob = self.dense2(prob)  # shape (batch_size, act_dim)
        
        # Instead of using tf.random.categorical (which is linked to memory leak issues),
        # implement the custom categorical sampling based on CDF searchsorted trick, as suggested in the issue.
        # cdf sampler expects logits, so convert prob (softmax output) back to logits using log.
        # But custom sampler expects logits.
        
        logits = tf.math.log(prob + 1e-20)  # avoid log(0) by small epsilon
        
        # Use the custom categorical sampler:
        # returns shape (batch_size, num_samples), here num_samples=1
        
        samples = categorical(logits, num_samples=1, dtype=tf.int64)
        
        # samples shape: (batch_size, 1)
        # squeeze last dimension for simplicity:
        samples = tf.squeeze(samples, axis=-1)  # shape (batch_size,)
        
        return samples

def categorical(logits, num_samples, dtype=None, seed=None):
    """
    Custom categorical sampler that avoids Gumbel trick induced memory leak
    Implements inverse CDF sampling given logits.
    
    Args:
        logits: Tensor of shape (batch, num_classes)
        num_samples: int, number of samples to draw per batch element
        dtype: output dtype of samples (defaults to tf.int64)
        seed: random seed (unused here, could be extended)
        
    Returns:
        Tensor of shape (batch, num_samples) of sampled class indices
    """
    logits = tf.convert_to_tensor(logits, name="logits")
    
    # For seed, using tf.compat.v1.get_seed to split seed -- preserve older seed mechanism.
    seed1, seed2 = tf.compat.v1.get_seed(seed)
    
    batch = tf.shape(logits)[0]
    num_classes = tf.shape(logits)[1]
    
    # Uniform samples for inverse CDF
    u = tf.random.uniform(shape=[batch, num_samples], seed=seed1, dtype=tf.float32)
    
    # Stability: subtract max logit to prevent overflow in exp
    max_logit = tf.reduce_max(logits, axis=1, keepdims=True)
    shifted_logits = logits - max_logit
    
    pdf = tf.exp(shifted_logits)  # unnormalized probabilities
    cdf = tf.cumsum(pdf, axis=1)
    
    cdf_last = tf.expand_dims(cdf[:, -1], axis=1)
    u_scaled = u * cdf_last  # scale uniform by total
        
    # Handle special cases: 
    # If num_samples==0 or batch==0, return zeros (shape: [batch, num_samples])
    cond = tf.logical_or(tf.equal(num_samples, 0), tf.equal(batch, 0))
    
    def true_fn():
        return tf.zeros([batch, num_samples], dtype=tf.int64 if dtype is None else dtype)
    
    def false_fn():
        # tf.searchsorted to find index where u_scaled fits into CDF
        return tf.searchsorted(cdf, u_scaled, side="right")
    
    samples = tf.cond(cond, true_fn, false_fn)
    
    if dtype is not None:
        samples = tf.cast(samples, dtype)
    else:
        samples = tf.cast(samples, tf.int64)
        
    return samples

def my_model_function():
    # Instantiate and return the MyModel instance
    return MyModel()

def GetInput():
    # Return random input tensor matching input shape expected by MyModel.
    # Based on usage in issue, input shape is (batch=500, features=500) with dtype float32.
    return tf.random.uniform(shape=(500, 500), dtype=tf.float32)

