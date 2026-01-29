# tf.random.uniform((B, num_visible), dtype=tf.float32) ← Input is a batch of binary visible units for an RBM with 'num_visible' features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_visible=784, num_hidden=500, mcsteps=5):
        """
        A simple RBM (Restricted Boltzmann Machine) model class implementing
        contrastive divergence (CD) steps as multiple Monte Carlo sampling layers.
        
        This reimplements the CD training logic as a Keras subclassed model,
        inspired by the original issue discussion.
        
        Assumptions:
        - Input is a batch of visible binary units of shape (B, num_visible)
        - Uses binary stochastic hidden units with sigmoid activations
        """
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.mcsteps = mcsteps
        
        # Weight matrix between visible and hidden units, initialized small random values
        self.W = tf.Variable(tf.random.normal(shape=(self.num_visible, self.num_hidden), stddev=0.01), name="W")
        # Biases for visible units
        self.bv = tf.Variable(tf.zeros(shape=(self.num_visible,)), name="bv")
        # Biases for hidden units
        self.bh = tf.Variable(tf.zeros(shape=(self.num_hidden,)), name="bh")
    
    def sample_prob(self, probs):
        """Sample binary states from probabilities."""
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))
    
    def call(self, inputs, training=None):
        """
        Forward pass computes one CD step chain of mcsteps Gibbs sampling.
        Returns the difference in expectation for the contrastive divergence gradient.
        
        inputs: batch of visible units (B, num_visible)
        """
        v0 = tf.cast(inputs, tf.float32)
        
        # Positive phase: Compute hidden probabilities and sample
        ph_mean = tf.nn.sigmoid(tf.matmul(v0, self.W) + self.bh)
        ph_sample = self.sample_prob(ph_mean)
        
        # Initialize vk and hk for Gibbs sampling
        vk = v0
        hk = ph_sample
        
        # Gibbs sampling for mcsteps
        for _ in range(self.mcsteps):
            # Sample visible units given hidden
            v_probs = tf.nn.sigmoid(tf.matmul(hk, tf.transpose(self.W)) + self.bv)
            vk = self.sample_prob(v_probs)
            # Sample hidden units given visible
            h_probs = tf.nn.sigmoid(tf.matmul(vk, self.W) + self.bh)
            hk = self.sample_prob(h_probs)
        
        # Negative phase
        # Compute gradients (contrastive divergence approximation)
        positive_grad = tf.matmul(tf.transpose(v0), ph_mean)
        negative_grad = tf.matmul(tf.transpose(vk), h_probs)
        
        # Contrastive divergence estimate (difference of outer products)
        cd_diff = positive_grad - negative_grad
        
        # For simplicity, output the loss proxy (free energy difference) or cd_diff norm can be used.
        # Here, we return the loss proxy: mean over batch of visible-negative free energy minus visible-positive free energy
        
        free_energy = lambda v: -tf.matmul(v, tf.expand_dims(self.bv, 1)) - \
                                tf.reduce_sum(tf.math.log(1 + tf.exp(tf.matmul(v, self.W) + self.bh)), axis=1, keepdims=True)
        
        fe_v0 = tf.squeeze(free_energy(v0))
        fe_vk = tf.squeeze(free_energy(vk))
        loss_proxy = tf.reduce_mean(fe_v0 - fe_vk)
        
        return loss_proxy

def my_model_function():
    """
    Return an instance of MyModel with default RBM architecture (typical for MNIST).
    """
    return MyModel(num_visible=784, num_hidden=500, mcsteps=5)

def GetInput():
    """
    Generate a random batch of binary visible units as input to MyModel.
    Assume batch size 32 and binary input modeled as Bernoulli with p=0.5.
    """
    batch_size = 32
    num_visible = 784
    # Generate random binary tensor (0 or 1)
    random_input = tf.cast(tf.random.uniform((batch_size, num_visible), 0, 2, dtype=tf.int32), tf.float32)
    return random_input

