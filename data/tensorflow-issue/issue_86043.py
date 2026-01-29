# tf.random.uniform((B, None, None, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The input is expected to be ragged tensors with shape (batch, ?, ?, 1)
        # Keras 3 tf.keras.layers.Resizing currently does not support RaggedTensor inputs
        # This is inferred from the issue discussion.
        # Here we implement a workaround by first converting ragged input to dense with padding,
        # then resizing to (10,10), global max pooling and softmax
        self.resize = tf.keras.layers.Resizing(10, 10)
        self.pool = tf.keras.layers.GlobalMaxPool2D()
        self.softmax = tf.keras.layers.Softmax()
        
    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        # inputs is a RaggedTensor with shape [B, None, None, 1]
        # We convert to dense with padding. This is necessary because
        # tf.keras.layers.Resizing does not support RaggedTensor as input
        # in Keras 3 (as per the issue's conclusion)
        dense_inputs = inputs.to_tensor(default_value=0)  # Pad to max dimensions in batch

        # Resize to (10, 10)
        resized = self.resize(dense_inputs)

        # Global max pool, resulting shape (B, channels)
        pooled = self.pool(resized)

        # Softmax over the last dimension, typically classes = 1 here (from example)
        # Note: The example warns that softmax on axis=-1 of shape (None, 1) will always be 1.
        # This behavior is preserved as is to reflect the original model.
        output = self.softmax(pooled)
        return output

def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Generate a batched RaggedTensor input matching the original scenario:
    # batch size = 4, each element is a ragged tensor of shape (x+1, x+1, 1)
    # where x ranges within the batch: e.g. lengths 1,2,3,4 etc.
    # We'll create ragged tensor with ragged rank 2 (height and width ragged dims).
    batch_size = 4

    ragged_values = []
    row_lengths = []
    col_lengths = []
    for i in range(batch_size):
        length = i + 1
        # Create dense tensor for each shape (length, length, 1)
        val = tf.random.uniform((length, length, 1), dtype=tf.float32)
        ragged_values.append(val)
        row_lengths.append(length)
        col_lengths.append(length)

    # Construct a RaggedTensor with ragged row and column dimensions.
    # To do this, flatten all values and create nested row splits.

    # Flatten each (length, length, 1) tensor to [(length*length), 1]
    flattened_values = [tf.reshape(v, (-1, 1)) for v in ragged_values]
    concatenated_values = tf.concat(flattened_values, axis=0)

    # Compute row splits for ragged dims
    # RaggedTensor with ragged_rank=2: ragged rows and columns
    # Outer ragged dim: batch dim
    # Inner ragged dim: row length per batch element
    # To get the row_splits for ragged dims:
    row_splits = [0]
    for length in row_lengths:
        row_splits.append(row_splits[-1] + length * length)  # total elements per example (flattened)
    # But this would mingle row and col dims into one ragged dim - instead, we need two ragged dims:
    # Since tf.RaggedTensor.from_tensor supports ragged_rank up to 2,
    # To create a ragged tensor representing (B, ragged_rows, ragged_cols, 1),
    # the cleanest way is to use RaggedTensor.from_tensor on a Python list:

    # Shorter approach for GetInput:
    # Just create a Python list of ragged tensors then construct a batch RaggedTensor:
    ragged_batch = tf.ragged.constant([
        tf.zeros((length, length, 1), dtype=tf.float32) for length in range(1, batch_size +1)
    ])

    # Replace zeros with uniform random:
    ragged_batch = tf.ragged.map_flat_values(tf.random.uniform, shape=(), dtype=tf.float32, ragged_tensor=ragged_batch)

    # This method respects ragged dimensions as per the example in the issue.
    return ragged_batch

