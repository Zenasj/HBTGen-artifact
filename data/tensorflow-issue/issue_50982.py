# tf.random.uniform((3, 3, 2), dtype=tf.float32) ← inferred from Z shape in the issue (batch=3, event=3, dim=2)

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct the chain of bijectors - here only a single Tanh bijector
        self.flow_chain = tfb.Chain(bijectors=[tfb.Tanh()])
        # Base distribution is 2-dimensional multivariate normal with diagonal covariance
        self.base_distribution = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])
        # Transformed distribution from base distribution passed through flow chain
        self.trans_dist = tfd.TransformedDistribution(distribution=self.base_distribution,
                                                      bijector=self.flow_chain)

    def call(self, inputs):
        """
        Forward pass returns log probability of inputs under the transformed distribution,
        i.e. log_prob of inputs under the normalizing flow model.
        
        inputs: Tensor of shape [batch, event, dim], exactly shape (3, 3, 2) inferred from example
        """
        # We must ensure event_ndims and input shapes are compatible with Bijector
        
        # From the issue: forward_min_event_ndims=0 in original flow
        # The error shown was an InvalidArgumentError related to broadcast shapes in inverse_log_det_jacobian.
        # The original call used event_ndims=2 in _forward_log_det_jacobian.
        # Tanh bijector expects event_ndims=1 for 2D vector inputs.
        # So adjusting event_ndims to 1 in bijector calls to match per-vector transformations.

        # Since the base distribution is 2D, the event shape is (2,), 
        # sample shape is (3,3) => inputs shape=(3,3,2).
        
        # Just use the TransformedDistribution's log_prob safely:
        # This will handle forward and jacobians internally with correct event_ndims.
        return self.trans_dist.log_prob(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    # Provide test input matching the expected shape in the example, dtype float32
    # Shape is batch_size=3, event_length=3, event_dim=2
    return tf.random.uniform(shape=(3, 3, 2), minval=-1.0, maxval=1.0, dtype=tf.float32)

