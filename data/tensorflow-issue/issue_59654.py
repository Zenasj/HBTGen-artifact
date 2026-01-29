# tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This MyModel serves as a minimal example model compatible with 
    TensorFlow 2.11+ and customizable optimizers such as the provided 
    Gravity optimizer. 
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple Conv2D layer as example
        self.conv = tf.keras.layers.Conv2D(8, kernel_size=3, padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)  # output 10 classes

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    """
    Return an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Return a random tensor matching input shape expected by MyModel.

    Assumption:
    - Input shape: batch size 1, 28x28 image, 3 channels (like colored image).
    This is arbitrary but reasonable for a convnet example.
    """
    return tf.random.uniform((1, 28, 28, 3), dtype=tf.float32)

# === The Gravity optimizer updated for TensorFlow 2.11+ ===

class Gravity(tf.keras.optimizers.Optimizer):
    """
    Custom Gravity optimizer updated for TF 2.11+.

    Notes:
    - Unlike tf.keras.optimizers.legacy.Optimizer, newer Optimizer API
      removes _set_hyper and replaces it by using self._hyperparameters dictionary 
      and public setters/getters.
    
    - We define hyperparameters as public attributes and use self._set_hyper
      replaced by self._hyperparameters update through super().__init__ call.

    References:
    - TensorFlow 2.11+ customization docs.
    """

    def __init__(self, learning_rate=0.1, alpha=0.01, beta=0.9, name="Gravity", **kwargs):
        # Define hyperparameters dictionary expected by base class
        hyper_params = dict(
            learning_rate=learning_rate,
            alpha=alpha,
            beta=beta
        )
        # Pass hyperparameters to base
        super().__init__(name=name, **kwargs)
        self._hyper['learning_rate'] = learning_rate
        self._hyper['alpha'] = alpha
        self._hyper['beta'] = beta
        self.epsilon = 1e-7

    def _create_slots(self, var_list):
        alpha = self._get_hyper("alpha")
        learning_rate = self._get_hyper("learning_rate")
        # Defensive: standard deviation for velocity initializer
        stddev = alpha / learning_rate if learning_rate != 0 else 1.0
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev)
        for var in var_list:
            self.add_slot(var, "velocity", initializer=initializer)

    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._get_hyper('learning_rate', var_dtype)
        beta = self._get_hyper("beta", var_dtype)
        t = tf.cast(self.iterations + 1, var_dtype)  # +1 to avoid zero division
        beta_hat = (beta * t + 1) / (t + 2)
        velocity = self.get_slot(var, "velocity")

        max_step_grad = 1.0 / tf.math.reduce_max(tf.math.abs(grad))
        gradient_term = grad / (1.0 + tf.math.square(grad / max_step_grad))

        updated_velocity = velocity.assign(beta_hat * velocity + (1 - beta_hat) * gradient_term)
        updated_var = var.assign_sub(lr_t * updated_velocity)

        return tf.group(updated_var, updated_velocity)

    def _resource_apply_sparse(self, grad, var, indices):
        # Sparse gradients not implemented in original; raise error.
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = super(Gravity, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "alpha": self._serialize_hyperparameter("alpha"),
            "beta": self._serialize_hyperparameter("beta"),
            "epsilon": self.epsilon,
        })
        return config

