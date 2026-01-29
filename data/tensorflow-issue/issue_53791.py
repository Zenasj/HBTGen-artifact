# tf.random.uniform((B, 20), dtype=tf.float32) ← inferred input shape from the example inputs in the issue (np.random.randn(1000, 20))
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use a Sequential stack of 3 Dense(64) layers + Dense(1) output as in the original example
        self.l = tf.keras.Sequential([
            tf.keras.layers.Dense(64) for _ in range(3)
        ] + [tf.keras.layers.Dense(1)])

    def call(self, inputs, training=None):
        # Use tf.summary.scalar inside call (not recommended for saving, but replicates the reported scenario)
        # Use tf.reduce_sum to summarize inputs as in example, scalar name 'avg_1'
        tf.summary.scalar(name='avg_1', data=tf.reduce_sum(inputs))
        return self.l(inputs)

def my_model_function():
    # Return an instance of MyModel as requested
    return MyModel()

def GetInput():
    # Create a random input tensor matching shape (batch_size, 20)
    # Using batch size 32 as a reasonable default for demonstration
    return tf.random.uniform((32, 20), dtype=tf.float32)

