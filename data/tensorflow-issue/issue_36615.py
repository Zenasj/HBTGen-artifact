# tf.random.uniform((B,), dtype=tf.int64)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # DenseHashTable requires explicit empty_key and deleted_key
        self.table = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.int64,
            default_value=-1,
            empty_key=0,
            deleted_key=-1)
        # Pre-insert some sample key-value pairs to simulate a built table
        # (Assuming this is a reasonable approach to enable saving/loading.)
        # These keys/values can be replaced as needed.
        self.table.insert([1, 2, 3, 4], [4, 3, 2, 1])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int64)])
    def call(self, input):
        # Lookup keys in DenseHashTable, returns values or default_value if not found.
        return self.table.lookup(input)

def my_model_function():
    # Return an instance of MyModel with the DenseHashTable initialized and prefilled.
    return MyModel()

def GetInput():
    # Return a random 1-D tensor of integers as input matching expected input signature.
    # Shape is (batch,), dtype int64. Let's use batch size 4 to match pre-inserted keys.
    return tf.random.uniform(shape=(4,), minval=0, maxval=10, dtype=tf.int64)

