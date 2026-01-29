# tf.random.uniform((None,)) ← Input shape inferred from the issue example generator yielding tf.zeros(2, 3),
# but shape is dynamic (None,), so we assume a 2D tensor with shape (2, 3) as the fixed input shape for clarity
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define two simple sub-models for demonstration.
        # Because the original issue is about tf.data.Dataset.from_generator memory leak,
        # and no explicit model was described, we'll simulate two models that
        # consume the input and compare outputs as an example fused model.
        self.model_a = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(5)
        ])
        self.model_b = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(5)
        ])

    def call(self, x):
        # x assumed shape: (batch_size, 2, 3) or (2, 3)
        # Flatten the first dimension if needed (if input is (2,3) -> batch_size=2)
        # We'll flatten batch and sequence dims for a general approach.
        # Although the original generator yields tf.zeros(2,3), 
        # here to keep compatibility, we treat input shape as (batch, 2, 3).
        # For this example, assume input shape is (batch, 2, 3).
        # Reshape input to (batch * 2, 3) to feed into Dense layer.
        shape = tf.shape(x)
        batch_dim = shape[0]
        seq_dim = shape[1]
        x_flat = tf.reshape(x, (batch_dim * seq_dim, 3))

        output_a = self.model_a(x_flat)
        output_b = self.model_b(x_flat)

        # Compare outputs: for instance, compute their elementwise absolute difference
        diff = tf.abs(output_a - output_b)

        # Aggregate difference per batch: sum over last axis
        diff_per_sample = tf.reduce_sum(diff, axis=-1)

        # Reshape back to (batch_dim, seq_dim)
        diff_per_sample = tf.reshape(diff_per_sample, (batch_dim, seq_dim))
        
        # Return the difference tensor as output - showing how the two sub-models compare
        return diff_per_sample

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel.
    # Since the original example generator outputs tf.zeros(2,3), batch size is unknown.
    # For this function, create a batch with batch size 4 of shape (4, 2, 3).
    # Using float32 as dtype.
    return tf.random.uniform((4, 2, 3), dtype=tf.float32)

