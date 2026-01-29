# tf.random.uniform((2, 2, 2, 2), dtype=tf.float32), tf.random.uniform((1, 1, 1, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the logs and error traces, the model must contain a Dense layer applied on a 4D input.
        # We'll create a small example to mimic the scenario that leads to the problematic matmul.
        # Assumption: The input shape inferred from call args is (2, 2, 2, 2).
        # We'll flatten the trailing dims to apply Dense and then reshape back.
        self.flatten = tf.keras.layers.Reshape(target_shape=(-1,))  # flatten last dims into vector
        self.dense = tf.keras.layers.Dense(units=4)  # arbitrary choice of units, matches some dims
        self.reshape_back = tf.keras.layers.Reshape(target_shape=(2, 2, 1))  # reshape output to a plausible shape

    def call(self, inputs, training=None):
        # inputs: tuple of tensors, as per call args in logs, e.g. two tensors
        # We'll only use the first input (which is 4D tensor) as the main data input for Dense application.
        x = inputs[0]
        # Flatten last dims except batch dim:
        batch_size = tf.shape(x)[0]

        # Flatten trailing spatial dims except batch dim to vectors, i.e. keep batch dim fixed
        x_flat = tf.reshape(x, [batch_size, -1])
        x_dense = self.dense(x_flat)
        # Reshape back to a 3D tensor for demonstration -- shape (batch_size, 2, 2, 1) was in logs
        # We reshape to (batch_size, 2, 2, 1) if possible; else fallback to (batch_size, -1, 1)
        try:
            x_out = tf.reshape(x_dense, [batch_size, 2, 2, 1])
        except:
            x_out = tf.expand_dims(x_dense, axis=-1)
        return x_out

def my_model_function():
    # Return an instance of MyModel
    # No special weights initialization is given or necessary.
    return MyModel()

def GetInput():
    # Based on logs, the model call receives two inputs:
    # - A 4D tensor of shape (2, 2, 2, 2), dtype float32
    # - A 4D tensor of shape (1, 1, 1, 1), dtype float32 (sometimes a 1D shape (1,))
    # We'll generate a tuple matching these inputs.

    tensor1 = tf.random.uniform((2, 2, 2, 2), dtype=tf.float32)
    tensor2 = tf.random.uniform((1, 1, 1, 1), dtype=tf.float32)
    return (tensor1, tensor2)

