# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape is unknown from issue, assume generic 4D tensor with float32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model implements a custom optimizer logic inspired by MyAdamOptimizer
        # Since the issue is about a missing _set_hyper and _get_hyper in TF 2.12,
        # here we demonstrate a model containing a custom training step that uses
        # manual hyperparameters instead of _set_hyper/_get_hyper.
        
        # For demonstration, define some trainable variables
        self.dense = tf.keras.layers.Dense(10)

        # Hyperparameters provided as attributes since _set_hyper/_get_hyper are no longer available
        self.learning_rate = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-7

        # Create slot variables for Adam optimizer manual implementation
        # slots for momentum (m) and variance (v) are initialized here for demonstration
        self.m = []
        self.v = []
        # Initialize these slots for each trainable variable; done on build or in first call
        self._slots_initialized = False

        # iteration count to simulate Adam step count
        self.iterations = tf.Variable(0, trainable=False, dtype=tf.int64)

    def build(self, input_shape):
        # Initialize slots m and v for self.dense weights and biases
        for var in self.dense.trainable_variables:
            m = self.add_weight(
                name=var.name.split(':')[0] + "_m",
                shape=var.shape,
                dtype=var.dtype,
                trainable=False,
                initializer="zeros"
            )
            v = self.add_weight(
                name=var.name.split(':')[0] + "_v",
                shape=var.shape,
                dtype=var.dtype,
                trainable=False,
                initializer="zeros"
            )
            self.m.append(m)
            self.v.append(v)
        self._slots_initialized = True
        super().build(input_shape)

    def call(self, inputs):
        return self.dense(inputs)

    @tf.function
    def train_step(self, inputs):
        # A simplified manual Adam optimizer step that uses the parameters as attributes
        if not self._slots_initialized:
            self.build(inputs.shape)

        with tf.GradientTape() as tape:
            predictions = self.call(inputs)
            # A dummy loss: mean squared error to zeros as placeholder
            loss = tf.reduce_mean(tf.square(predictions))

        trainable_vars = self.dense.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        # Adam update manually, simplified version
        self.iterations.assign_add(1)
        lr = tf.cast(self.learning_rate, grads[0].dtype)
        beta_1 = tf.cast(self.beta_1, grads[0].dtype)
        beta_2 = tf.cast(self.beta_2, grads[0].dtype)
        epsilon = tf.cast(self.epsilon, grads[0].dtype)

        for i, (grad, var) in enumerate(zip(grads, trainable_vars)):
            m = self.m[i]
            v = self.v[i]

            # Update biased first moment estimate
            m.assign(beta_1 * m + (1 - beta_1) * grad)
            # Update biased second raw moment estimate
            v.assign(beta_2 * v + (1 - beta_2) * tf.square(grad))

            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - tf.pow(beta_1, tf.cast(self.iterations, tf.float32)))
            # Compute bias-corrected second moment estimate
            v_hat = v / (1 - tf.pow(beta_2, tf.cast(self.iterations, tf.float32)))

            var_update = lr * m_hat / (tf.sqrt(v_hat) + epsilon)

            var.assign_sub(var_update)

        return loss

def my_model_function():
    return MyModel()

def GetInput():
    # The issue context is about optimizer, but model input shape isn't defined
    # For demonstration, assume input is a batch of vectors of size 20 (shape: [B, 20])
    batch_size = 4  # arbitrary batch size
    input_dim = 20
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

