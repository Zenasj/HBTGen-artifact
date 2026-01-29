# tf.random.uniform((64, 28*28), dtype=tf.float32) ← Input shape inferred from RBM batch_size=64, nv=28*28 flattened image vector

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, nv=28 * 28, nh=512, cd_steps=3):
        super().__init__()
        # Seed for reproducibility
        tf.random.set_seed(42)

        self.nv = nv  # number of visible units (input dim)
        self.nh = nh  # number of hidden units
        self.cd_steps = cd_steps  # number of contrastive divergence steps

        # Initialize weights and biases similar to RBM original code
        # Using tf.Variable for weights and biases to replicate RBM parameters
        self.W = tf.Variable(
            tf.random.truncated_normal((nv, nh), stddev=0.01, seed=42),
            name='W')
        self.bv = tf.Variable(tf.zeros([nv, 1]), name='bv')  # visible bias
        self.bh = tf.Variable(tf.zeros([nh, 1]), name='bh')  # hidden bias

    def bernoulli_sample(self, p):
        # Bernoulli sampling: output 1 with probability p, else 0
        # Using tf.sign and uniform random tensor as in original code
        # tf.sign(p - random_uniform) gives +1 if p > uniform, 0 if equal, -1 if less;
        # tf.nn.relu zeroes negative => binary samples 0/1
        uniform_sample = tf.random.uniform(tf.shape(p), dtype=p.dtype)
        return tf.nn.relu(tf.sign(p - uniform_sample))

    def sample_h(self, v):
        # Sample hidden units given visible
        # ph_given_v = sigmoid(v * W + bh)
        linear_part = tf.matmul(v, self.W) + tf.squeeze(self.bh)  # shape [batch, nh]
        ph_given_v = tf.sigmoid(linear_part)
        return self.bernoulli_sample(ph_given_v)

    def sample_v(self, h):
        # Sample visible units given hidden
        # pv_given_h = sigmoid(h * W^T + bv)
        linear_part = tf.matmul(h, tf.transpose(self.W)) + tf.squeeze(self.bv)  # shape [batch, nv]
        pv_given_h = tf.sigmoid(linear_part)
        return self.bernoulli_sample(pv_given_h)

    def energy(self, v):
        # Energy function of RBM: E(v) = - sum_i b_i v_i - sum_j log(1 + exp(sum_i v_i W_ij + bh_j))
        # v shape: [batch, nv]
        # b_term: batch x 1 
        # h_term: batch x 1 (sum over hidden units)
        b_term = tf.matmul(v, self.bv)  # shape [batch, 1]
        linear_transform = tf.matmul(v, self.W) + tf.squeeze(self.bh)  # shape [batch, nh]
        h_term = tf.reduce_sum(tf.math.log(tf.exp(linear_transform) + 1), axis=1, keepdims=True)  # shape [batch,1]
        energy_per_sample = -h_term - b_term  # shape [batch,1]
        mean_energy = tf.reduce_mean(energy_per_sample)
        return mean_energy

    def gibbs_step(self, k, vk):
        # Perform one step of Gibbs sampling: v -> h -> v
        hk = self.sample_h(vk)
        vk = self.sample_v(hk)
        return k + 1, vk

    @tf.function
    def call(self, input_tensor):
        # Forward pass: input_tensor shape: [batch, nv]
        # For demonstration, run CD steps and return energy difference (loss)
        v = tf.round(input_tensor)  # binarize input (like placeholder rounding in original)
        vk = tf.identity(v)

        i = tf.constant(0)
        k = tf.constant(self.cd_steps)

        # Since tf.while_loop requires a cond function, define it here
        def cond(k_, vk_):
            return k_ < self.cd_steps

        def body(k_, vk_):
            k_new, vk_new = self.gibbs_step(k_, vk_)
            return k_new, vk_new

        k_final, vk_final = tf.while_loop(cond, body, loop_vars=[i, vk])

        # Compute contrastive divergence loss (energy difference)
        loss = self.energy(v) - self.energy(vk_final)
        # Output the loss as forward output for convenience
        return loss

def my_model_function():
    # Instantiate model with default parameters (visible units=28*28, hidden=512, cd_steps=3)
    return MyModel()

def GetInput():
    # Return a random uniform tensor input matching (batch_size=64, nv=28*28)
    # dtype float32 consistent with RBM input expected
    batch_size = 64
    nv = 28 * 28
    return tf.random.uniform((batch_size, nv), minval=0.0, maxval=1.0, dtype=tf.float32)

