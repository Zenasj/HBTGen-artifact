# tf.raw_ops.UpperBound(sorted_inputs=x, values=int16_tensor) with 
# input shapes: sorted_inputs=(10, 8), values=(10, 8) (inferred from usage in code)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function(jit_compile=True)
    def call(self, x):
        # Generate a random int16 tensor matching expected shape for values input
        random_tensor = tf.random.uniform([10, 8], minval=-32768, maxval=32767, dtype=tf.int32)
        int16_tensor = tf.cast(random_tensor, tf.int16)

        # Perform UpperBound operation (expects sorted_inputs and values)
        # x is assumed to be a sorted int16 tensor of shape (10, 8)
        upper_bound_result = tf.raw_ops.UpperBound(sorted_inputs=x, values=int16_tensor)

        return upper_bound_result

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random int16 tensor simulating the "sorted_inputs" input of shape (10, 8)
    # Since UpperBound expects sorted inputs, we sort it along axis=-1 to approximate
    random_tensor = tf.random.uniform([10, 8], minval=-32768, maxval=32767, dtype=tf.int32)
    int16_tensor = tf.cast(random_tensor, tf.int16)
    sorted_int16 = tf.sort(int16_tensor, axis=-1)
    return sorted_int16

