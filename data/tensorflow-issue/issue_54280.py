# tf.random.uniform((50,)) ← Input shape is (50,), a 1D tensor of 50 floats as per example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single dense layer as per the example Sequential model
        self.dense = tf.keras.layers.Dense(1, input_shape=(1,))

        # We keep an optimizer attribute to emulate the compile/recompile scenario
        # Use default SGD without learning rate initially
        self.optimizer = tf.keras.optimizers.SGD()
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, training=False):
        # Forward pass: dense layer output
        return self.dense(inputs)

    def compile_model(self, learning_rate=None):
        # Method to set optimizer with learning_rate, mimicking .compile()
        if learning_rate is None:
            self.optimizer = tf.keras.optimizers.SGD()
        else:
            # We force the learning rate dtype to float32 to mimic the fix for the issue
            # The issue was that using a Python float results in float64, causing loading errors.
            # So we convert learning_rate to a tf.float32 scalar explicitly.
            # This corresponds to the workaround commented in issue:
            # optimizer=tf.keras.optimizers.SGD(learning_rate=float(...))  # workaround
            if isinstance(learning_rate, float):
                lr = tf.constant(learning_rate, dtype=tf.float32)
            elif isinstance(learning_rate, tf.Tensor):
                lr = tf.cast(learning_rate, tf.float32)
            else:
                lr = tf.constant(float(learning_rate), dtype=tf.float32)
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        # Loss fn stays the same, mse

    @tf.function(jit_compile=True)
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.call(x, training=True)
            loss = self.loss_fn(y, predictions)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

def my_model_function():
    # Return an instance of MyModel with default initialization
    model = MyModel()
    model.compile_model()
    return model

def GetInput():
    # Return a random uniform tensor of shape (50, 1) to match Dense layer input_shape=(1,)
    # Input in the example is 1D data points, so shape (50,1) as batch of 50 scalar values.
    return tf.random.uniform((50, 1), dtype=tf.float32)

