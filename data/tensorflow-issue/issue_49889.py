# tf.random.uniform((5, 1, 2, 4), dtype=tf.float32) ← inferred input shape from example tensor x

import tensorflow as tf

class OptimizerWrapper(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, name=None, **kwargs):
        super(OptimizerWrapper, self).__init__(name, **kwargs)
        self._optimizer = optimizer

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # apply_state is an optional argument in newer TF versions required to avoid colocation errors
        if apply_state is not None:
            return self._optimizer._resource_apply_dense(grad, var, apply_state)
        else:
            return self._optimizer._resource_apply_dense(grad, var)

    def _resource_apply_sparse(self, grad, var, apply_state=None):
        if apply_state is not None:
            return self._optimizer._resource_apply_sparse(grad, var, apply_state)
        else:
            return self._optimizer._resource_apply_sparse(grad, var)

    def _prepare(self, var_list):
        # Delegating _prepare to underlying optimizer to handle apply_state properly
        return self._optimizer._prepare(var_list)

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        # Delegate apply_gradients - optional but recommended to avoid errors
        return self._optimizer.apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

    def get_config(self):
        return self._optimizer.get_config()


class SimplePiecewiseConstantDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, boundaries, values, name=None):
        super(SimplePiecewiseConstantDecay, self).__init__()
        if len(boundaries) != len(values) - 1:
            raise ValueError(
                "The length of boundaries should be 1 less than the length of values"
            )
        self.boundaries = boundaries
        self.values = values
        self.name = name

    def __call__(self, step):
        # Using tf.cond to avoid tf.case and device colocation issues on GPU
        def case0():
            return tf.constant(self.values[0], dtype=tf.float32)

        def case1_2():
            cond = tf.greater(step, self.boundaries[-1])
            return tf.cond(cond,
                           lambda: tf.constant(self.values[-1], dtype=tf.float32),
                           lambda: tf.constant(self.values[0], dtype=tf.float32))

        return tf.cond(tf.less_equal(step, self.boundaries[0]), case0, case1_2)

    def get_config(self):
        return {
            "boundaries": self.boundaries,
            "values": self.values,
            "name": self.name
        }


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple sequential model with one Dense layer as per original example
        self.dense = tf.keras.layers.Dense(8)

        # Define piecewise constant decay learning rate function
        boundaries = [100000, 110000]
        values = [1.0, 0.5, 0.1]
        # Use the custom SimplePiecewiseConstantDecay schedule to avoid tf.case device errors
        self.learning_rate_fn = SimplePiecewiseConstantDecay(boundaries, values)

        # Base optimizer SGD with momentum and custom learning rate schedule
        base_opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate_fn, momentum=1.0)
        # Wrap the base optimizer in the OptimizerWrapper to simulate user's custom wrapper
        self.optimizer = OptimizerWrapper(base_opt)

    @tf.function
    def call(self, inputs):
        return self.dense(inputs)

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            y = self.call(x)
            loss = tf.reduce_mean(y)

        grads = tape.gradient(loss, self.trainable_variables)
        # Apply gradients using wrapped optimizer
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


def my_model_function():
    # Instantiate and return the MyModel instance
    return MyModel()


def GetInput():
    # Return a batch of random input tensors matching the shape used in example (5,1,2,4)
    # Use float32 dtype as typical for Keras models
    return tf.random.uniform((5, 1, 2, 4), dtype=tf.float32)

