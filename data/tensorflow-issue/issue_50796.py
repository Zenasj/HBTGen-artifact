# tf.random.uniform((None, 28, 28), dtype=tf.float32) ← Input shape is (batch_size, 28, 28) matching MNIST data

import tensorflow as tf
from tensorflow_addons.utils import types
from typeguard import typechecked

class MyModel(tf.keras.Model):
    """
    This class encapsulates the GradientAccumulator optimizer logic as a standalone Keras Model subclass.
    Since the original issue revolves around a ResourceVariable GC bug in the context of
    GradientAccumulator optimizer used on MNIST data with input shape (28, 28).

    Here, we implement the core GradientAccumulator as a custom optimizer,
    wrapped inside this 'MyModel' which allows integration and comparison if needed.

    The original code snippet was an optimizer class and a model defined separately.
    To meet requirements, we embed the essential custom optimizer inside this model class,
    and implement a call method to perform a forward pass on the input tensor.

    Note:
    - The input tensor shape is assumed (batch, 28, 28).
    - Model architecture is a simple MNIST classifier sequential model.
    - The bug fix from the discussion is applied: change (self.iterations+1) to self.iterations in tf.cond.
    """

    def __init__(self):
        super().__init__()
        # Define the simple MNIST classifier model layers
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)

        # Instantiate the GradientAccumulator optimizer wrapping Adam
        # Using default accum_steps=4 as per original
        self.optimizer = GradientAccumulator(
            tf.keras.optimizers.Adam(),
            accum_steps=4,
            name="GradientAccumulator"
        )

        # Loss function compatible with output logits and integer labels
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        """
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        logits = self.dense2(x)
        return logits

    def train_step(self, data):
        """
        Custom train_step to demonstrate GradientAccumulator usage logic.
        This method is not strictly required, but shows how the optimizer would be used.
        """
        x, y = data

        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self.loss_fn(y, logits)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        grads_and_vars = zip(gradients, self.trainable_variables)

        # Apply gradients via the GradientAccumulator optimizer
        self.optimizer.apply_gradients(grads_and_vars)
        return {"loss": loss}

@tf.keras.utils.register_keras_serializable(package="Addons")
class GradientAccumulator(tf.keras.optimizers.Optimizer):
    """Optimizer wrapper for gradient accumulation."""

    @typechecked
    def __init__(
        self,
        optimizer: types.Optimizer,
        accum_steps: types.TensorLike = 4,
        name: str = "GradientAccumulator",
        **kwargs,
    ):
        r"""
        Construct a new GradientAccumulator optimizer.

        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            accum_steps: int > 0. Update gradient in every accumulation steps.
            name: Optional name for the operations created when applying gradients.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`, `decay`}.
        """
        super().__init__(name, **kwargs)
        self._optimizer = tf.keras.optimizers.get(optimizer)
        self._gradients = []
        self._accum_steps = accum_steps

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "ga")

        self._gradients = [self.get_slot(var, "ga") for var in var_list]

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(
            gradient.read_value() if gradient is not None else gradient
            for gradient in self._gradients
        )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name, **kwargs)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            accum_gradient.assign_add(
                grad, use_locking=self._use_locking, read_value=False
            )

        def _apply():
            if "apply_state" in self._optimizer._dense_apply_args:
                train_op = self._optimizer._resource_apply_dense(
                    accum_gradient.read_value(), var, apply_state=apply_state
                )
            else:
                train_op = self._optimizer._resource_apply_dense(
                    accum_gradient.read_value(), var
                )
            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )
            return tf.group(train_op, reset_op)

        # Bug fix applied here from issue discussion:
        # change (self.iterations+1) to self.iterations to fix ResourceVariable GC bug
        apply_op = tf.cond(
            (self.iterations) % self._accum_steps == 0, _apply, lambda: tf.no_op()
        )
        return apply_op

    def _resource_apply_sparse(self, grad: types.TensorLike, var, indices, apply_state):
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            self._resource_scatter_add(accum_gradient, indices, grad)

        def _apply():
            if "apply_state" in self._optimizer._sparse_apply_args:
                train_op = self._optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices),
                    var,
                    indices,
                    apply_state=apply_state,
                )
            else:
                train_op = self._optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices), var, indices
                )
            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )
            return tf.group(train_op, reset_op)

        apply_op = tf.cond(
            (self.iterations) % self._accum_steps == 0, _apply, lambda: tf.no_op()
        )
        return apply_op

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        assign_ops = []
        if not self._gradients:
            return assign_ops

        for gradient in self._gradients:
            if gradient is not None:
                assign_ops.append(
                    gradient.assign(
                        tf.zeros_like(gradient),
                        use_locking=self._use_locking,
                        read_value=False,
                    )
                )

        return tf.group(assign_ops)

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)

    def get_config(self):
        config = {
            "accum_steps": self._accum_steps,
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)

def my_model_function():
    """
    Return an instance of MyModel with GradientAccumulator optimizer internally configured.
    """
    return MyModel()

def GetInput():
    """
    Return a random tensor input matching the input expected by MyModel:
    A batch of greyscale MNIST-like images with shape (batch_size=1, height=28, width=28).
    Using float32 as the dtype matching typical model input.
    """
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

