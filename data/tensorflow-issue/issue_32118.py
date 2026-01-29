# tf.random.uniform((K, 10000), dtype=tf.float64)  ← Here K=10 (number of replicate datasets), 10000 datapoints each

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model encapsulates multiple independent optimizers, each optimizing 
    a scalar variable x to minimize mean squared error (MSE) between the variable 
    and a bootstrap replicate dataset.

    We instantiate one optimizer per replicate dataset and perform fixed-step SGD 
    optimization inside each call. This matches the workaround approach in the issue.

    Forward pass accepts a tensor of shape (K, 10000), 
    where K is number of replicates, and returns a tensor of the optimized x values 
    (one per replicate).

    This design avoids dynamic variable creation inside tf.function by creating
    separate optimizer instances upfront for each replicate.
    """

    def __init__(self, num_replicates=10, num_iters=100, learning_rate=1.0):
        super().__init__()

        self.num_replicates = num_replicates
        self.num_iters = num_iters
        self.learning_rate = learning_rate

        # Create list of variables and optimizers, one per replicate
        self.variables = []
        self.optimizers = []
        for i in range(self.num_replicates):
            # Each variable is scalar float64 initialized to zero
            x = tf.Variable(0., dtype=tf.float64, name=f"x_{i}")
            opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            # Initialize optimizer variables by running one dummy apply_gradients call:
            # SGD optimizer in TF 2.x creates iteration counters as variables lazily,
            # so we force creation here.
            opt._create_all_weights(x)
            self.variables.append(x)
            self.optimizers.append(opt)

    @tf.function
    def call(self, replicates):
        """
        Args:
            replicates: tf.Tensor of shape (K, 10000) dtype float64,
                        containing bootstrap replicate datasets.

        Returns:
            tf.Tensor of shape (K,) dtype float64,
            the optimized scalar x for each replicate dataset.
        """
        results = []

        # Iterate over replicates and optimize their corresponding variable x
        for i in tf.range(self.num_replicates):
            x = self.variables[i]
            opt = self.optimizers[i]
            data = replicates[i]

            # Reset variable to zero at start of optimization
            x.assign(0.)

            # Reset optimizer state variables (e.g. iteration count) - assume single var
            for v in opt.variables():
                v.assign(0)

            # Perform fixed number of gradient descent steps to minimize MSE
            for _ in tf.range(self.num_iters):
                with tf.GradientTape() as tape:
                    # Objective: mean squared error between data and x scalar
                    obj = tf.reduce_mean((data - x) ** 2)
                grad = tape.gradient(obj, x)
                opt.apply_gradients([(grad, x)])

            results.append(x)

        # Stack results into a 1D tensor shape (K,)
        return tf.stack(results)

def my_model_function():
    """
    Returns:
        Instance of MyModel with default parameters.
    """
    return MyModel()

def GetInput():
    """
    Returns:
        A tf.Tensor matching input expected by MyModel.call():
        shape (K=10, 10000), dtype tf.float64, random normal values.
    """
    K = 10
    N = 10000
    # Generate random normal bootstrap replicates
    base_data = tf.random.normal((N,), dtype=tf.float64)
    # Sample K bootstrap replicates with replacement (simulate with tf.random.uniform)
    # Here we simulate bootstrap sample by random indices from base_data for each replicate.
    idx = tf.random.uniform(shape=(K, N), maxval=N, dtype=tf.int32)
    replicates = tf.gather(base_data, idx, batch_dims=1)
    return replicates

