# tf.random.uniform((batch_size, max_num_coords, in_channels), dtype=tf.float32), indices: (batch_size, max_num_coords, 2), num_valid_coordinates: (batch_size,)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, dense_shape=(28, 28)):
        super().__init__()
        # assuming dense_shape (image_height, image_width) as 28x28 for example
        self.dense_shape = dense_shape

    @tf.function
    def sparse_tensor_fn(self, indices, num_valid_coordinates, padded_features, i):
        # Build a SparseTensor for batch item i
        # indices[i] shape: [max_num_coords_per_batch, 2]
        # We take only the first num_valid_coordinates[i] valid coords and features
        idx = indices[i][:num_valid_coordinates[i]]  # shape: [num_valid_coords_i, 2]
        vals = padded_features[i][:num_valid_coordinates[i]]  # shape: [num_valid_coords_i, in_channels]
        # squeeze if in_channels=1 to 1D values, else keep as is by flattening channels
        # To keep consistent with tf.sparse.SparseTensor values shape which is 1-D,
        # we flatten the channel dimension and multiply coords accordingly if in_channels>1.
        # For simplicity, we assume in_channels=1 here, as original code has no channel info on output.
        vals = tf.squeeze(vals, axis=-1)  # from shape [N,1] -> [N]
        sparse_tensor = tf.sparse.SparseTensor(indices=idx,
                                               values=vals,
                                               dense_shape=[self.dense_shape[0], self.dense_shape[1]])
        return sparse_tensor

    @tf.function
    def dense_from_sparse(self, inputs):
        # inputs is a tuple: (indices, num_valid_coordinates, padded_features)
        indices, num_valid_coordinates, padded_features = inputs
        batch_size = tf.shape(padded_features)[0]

        # Use tf.map_fn over batch dimension to create SparseTensors
        sparse_tensors = tf.map_fn(
            lambda i: self.sparse_tensor_fn(indices, num_valid_coordinates, padded_features, i),
            tf.range(batch_size),
            dtype=tf.sparse.SparseTensor
        )

        # tf.map_fn produces a RaggedTensor of SparseTensors, we flatten and convert each sparse tensor to dense
        # But tf.map_fn with SparseTensor output creates a Tensor with dtype=Variant (opaque dtype)
        # Hence, we convert sparse tensors to dense individually in the same map_fn
        dense_batch = tf.map_fn(
            lambda sp: tf.sparse.to_dense(sp, default_value=0.0),
            sparse_tensors,
            dtype=tf.float32
        )
        return dense_batch

    def call(self, inputs):
        # inputs tuple of (indices, num_valid_coordinates, padded_features)
        return self.dense_from_sparse(inputs)


def my_model_function():
    # Example dense_shape is commonly 28x28, user can customize if desired here.
    return MyModel(dense_shape=(28, 28))

def GetInput():
    """
    Construct a sample input tuple (indices, num_valid_coordinates, padded_features) compatible with MyModel.
    Assumptions:
    - batch_size=2
    - max_num_coords_per_batch=5
    - in_channels=1 (for simplicity)
    - coordinates are within 28x28 image
    """
    batch_size = 2
    max_num_coords = 5
    in_channels = 1

    # indices shape: [batch_size, max_num_coords, 2], tf.int64 or tf.int32
    indices = tf.constant([
        [[0, 0], [1, 1], [2, 2], [0, 0], [0, 0]],  # padding repeats (0,0) for example
        [[3, 3], [4, 4], [5, 5], [6, 6], [0, 0]]
    ], dtype=tf.int64)

    # num_valid_coordinates shape: [batch_size]
    num_valid_coordinates = tf.constant([3, 4], dtype=tf.int32)

    # padded_features shape: [batch_size, max_num_coords, in_channels]
    padded_features = tf.constant([
        [[1.0], [2.0], [3.0], [0.0], [0.0]],
        [[4.0], [5.0], [6.0], [7.0], [0.0]]
    ], dtype=tf.float32)

    return (indices, num_valid_coordinates, padded_features)

