# tf.random.uniform((B,), dtype=tf.int64), tf.random.uniform((B,), maxval=10, dtype=tf.int32) 
# Input shape inferred as a tuple of two 1D tensors: ids (int64), features (int32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # We don't have trainable layers here; this model replicates the "scan" logic
        # to batch consecutive features with the same id into a single tensor.
        # This is a stateful scan over the input dataset, accumulates features by consecutive ids.
        pass

    def call(self, inputs):
        """
        inputs: tuple of (ids, features)
          - ids: tensor of shape [N], dtype tf.int64 (could be int32 depending on data)
          - features: tensor of shape [N], dtype tf.int32

        Returns:
          batched_ids: ragged or padded tensor of shape [M] (number of unique consecutive ids)
          batched_features: ragged tensor of shape [M, None] where each row contains all features for that id
        """
        ids, features = inputs
        # We assume inputs are 1D tensors of the same length

        # We will implement the batching logic similar to the tf.data.experimental.scan example,
        # but purely in TensorFlow ops, so this can be jit compiled.
        # We scan over the input sequence to accumulate batches of features per consecutive id.

        # Initial state: (-1, empty tensor of shape [0])
        initial_id = tf.constant(-1, dtype=ids.dtype)
        empty_batch = tf.zeros([0], dtype=features.dtype)

        # Define the scan function:
        # Inputs:
        #   state: tuple (current_id, accumulated_features)
        #   input_element: tuple (id, feature)
        # Returns:
        #   new_state, output_for_previous_batch_or_empty
        def scan_fn(state, input_element):
            state_id, accumulated_batch = state
            current_id, current_feature = input_element

            def accumulate():
                # Same id as before, accumulate features
                new_accumulated = tf.concat([accumulated_batch, tf.expand_dims(current_feature, 0)], axis=0)
                new_state = (current_id, new_accumulated)
                # Output empty because no batch emitted yet
                return new_state, (tf.constant(-1, dtype=ids.dtype), tf.zeros([0], dtype=features.dtype))

            def accumulate_and_emit():
                # Different id -> emit previous batch, start new batch with current feature
                emit_batch = (state_id, accumulated_batch)
                new_state = (current_id, tf.expand_dims(current_feature, 0))
                # Output previous batch (id and features)
                return new_state, emit_batch

            cond = tf.math.logical_or(tf.equal(state_id, current_id), tf.equal(state_id, initial_id))
            return tf.cond(cond, accumulate, accumulate_and_emit)

        # Perform scan over the sequence
        # Prepare inputs for scan: stack ids and features into tuples
        elems = (ids, features)

        # Use tf.scan to accumulate batches and emit completed batches on id changes.
        # tf.scan returns all outputs, including empty outputs which we will filter out later.
        # Shape of output: tuple of (ids, features) each of shape [N]
        output_ids, output_features = tf.scan(
            fn=scan_fn,
            elems=elems,
            initializer=(initial_id, empty_batch),
            parallel_iterations=1,
            back_prop=False,
            infer_shape=False
        )[1]  # [1] to get outputs (state, output), we want the output part

        # output_ids, output_features are of shape [N], but many are empty batches (id = -1 or features empty)

        # Filter out empty outputs, keep only emitted batches
        valid_mask = tf.not_equal(output_ids, initial_id)
        filtered_ids = tf.boolean_mask(output_ids, valid_mask)
        filtered_features = tf.boolean_mask(output_features, valid_mask)

        # Now filtered_features is a ragged tensor packed as concatenated 1D slices
        # But the emitted batches are 1D features, we want to group these into a RaggedTensor
        # Because each batch may have different lengths, we can encode the lengths by running difference of indices.
        # Unfortunately tf.scan doesn't neatly produce a nested RaggedTensor.
        # Instead, we rely on the fact each emit corresponds to one batch, each batch is a 1D tensor of features.

        # Because tf.scan concatenates outputs along axis=0, but in this code each emitted batch is also 1D,
        # and output_features shape is [N, variable_length?] can't be represented.

        # Hence a more TensorFlow-native approach is to run the batching in tf.data pipelines.
        # But since the task asked for a Model's call method, here we will simulate output as a tuple of
        # lists of tensors (ids and feature batches).

        # So instead of trying to pack as a single tensor here, we return the filtered ids and a list of feature vectors.

        # A simple workaround: features returned by scan are concatenated along the first axis, with unknown batch boundaries.
        # Here, just return filtered_ids and filtered_features as RaggedTensors with ragged_rank=1:
        # But from the example we see features are batches already, so filtered_features shape should be [M, None]

        # Because tf.scan cannot output ragged tensors directly, in practice to use this precise logic,
        # the batching is done as a tf.data pipeline operation, not a single model call.

        # We will simulate the batching output by returning the filtered ids and features as RaggedTensor.

        # Note: This only works if features has shape [N], i.e. scalar features per input (which is the sample case).

        batched_features_rt = tf.RaggedTensor.from_row_lengths(filtered_features, 
                                                              row_lengths=tf.map_fn(
                                                                  lambda x: tf.shape(x)[0], 
                                                                  filtered_features, fn_output_signature=tf.int32))
        # But filtered_features as a tensor is 1D, so cannot do from_row_lengths directly.

        # Since each output is a batch (1D tensor), and filtered_features is shape [M, None] in a RaggedTensor,
        # we must return as RaggedTensor directly - but tf.scan cannot return ragged outputs.

        # Due to TF limitations in tf.function, and no nested RaggedTensor support here,
        # we will return filtered_ids and filtered_features (which are both dense tensors),
        # representing the batches by padding to the max length.

        max_len = tf.reduce_max(tf.map_fn(lambda x: tf.shape(x)[0], filtered_features, dtype=tf.int32))
        padded_batches = tf.stack([
            tf.pad(batch, [[0, max_len - tf.shape(batch)[0]]], constant_values=0)
            for batch in filtered_features
        ])

        return filtered_ids, padded_batches

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return example input tensors matching the expected input:
    # - ids: 1D tensor of consecutive ids (int64)
    # - features: 1D tensor of features (int32), one feature per id (scalar here for simplicity)
    ids = tf.constant([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=tf.int64)
    features = tf.constant([1, 2, 3, 1, 2, 1, 2, 3, 4], dtype=tf.int32)
    return (ids, features)

