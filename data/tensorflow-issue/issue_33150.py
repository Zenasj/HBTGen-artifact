# tf.random.uniform((B, D_in), dtype=tf.float32) ← Assuming input is a batch of sequences with embedding dimension D_in

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Minimal example model to recreate issue and demonstrate fix/workaround:
        # A simple dense model + Adam optimizer with tracked variables for beta_1, beta_2.
        #
        # This reflects the examples seen in the issue discussion, where Adam optimizer's
        # attributes need to be tf.Variable to avoid checkpoint unresolved warnings.

        self.dense = tf.keras.layers.Dense(5)
        # Using modified Adam optimizer with tf.Variable attributes for betas.
        # This is a workaround from the issue discussion gist:
        # https://gist.github.com/yoshihikoueno/4ff0694339f88d579bb3d9b07e609122
        # to convert float betas to tf.Variables so checkpoint loads patches properly.

        # Custom Adam optimizer subclass with Variables tracking beta_1 and beta_2
        class TrackedAdam(tf.keras.optimizers.Adam):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Convert beta_1 and beta_2 to tf.Variable to track in checkpoint
                # Only if they are floats, to avoid reassigning multiple times.
                if isinstance(self._beta_1, float):
                    self._beta_1 = tf.Variable(self._beta_1, trainable=False, dtype=tf.float32, name="beta_1")
                if isinstance(self._beta_2, float):
                    self._beta_2 = tf.Variable(self._beta_2, trainable=False, dtype=tf.float32, name="beta_2")

            # Override _get_hyper to handle tf.Variable betas — required for TF2.0 compatibility
            def _get_hyper(self, name, dtype=None):
                if name == 'beta_1':
                    return tf.convert_to_tensor(self._beta_1, dtype=dtype)
                if name == 'beta_2':
                    return tf.convert_to_tensor(self._beta_2, dtype=dtype)
                return super()._get_hyper(name, dtype=dtype)

        self.optimizer = TrackedAdam(learning_rate=0.1)
        # The learning rate is set high so that optimizer variables get created quickly when
        # the model is trained on a dummy example.

        # To avoid unresolved checkpoint variables warnings, the model MUST be "built" before restoring.
        # So calling model on an example input or compiling+training on dummy data is often needed.

    def call(self, inputs):
        # Simple forward that applies the dense layer (no activation needed)
        return self.dense(inputs)

def my_model_function():
    # Return a new instance of MyModel.
    # Note: The user must "build" this model or call it on inputs before restoring weights,
    # to ensure any optimizer variables (like iter, m, v) get properly created and tracked.

    model = MyModel()
    return model

def GetInput():
    # Return a dummy input tensor compatible with model's expected input.
    # In the issue, the example_x was tf.constant([[1.]]) shaped (1,1).
    # For generality, produce a batch of one example with shape (1,1), dtype float32.
    return tf.random.uniform((1, 1), dtype=tf.float32)

