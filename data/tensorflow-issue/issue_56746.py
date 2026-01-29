# tf.random.uniform((N/A)) ← This issue is about tf.data.Dataset from_generator interleaving, no fixed input shape inferred

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No actual trainable model, this replicates the workaround to interleave generators correctly.

    def call(self, inputs):
        # inputs is expected to be ignored here, but kept for compatible call signature
        # We'll create and interleave datasets similarly to the workaround pattern shared.
        N_DATASETS_TO_INTERLEAVE = 10

        def hello(idx):
            # Generator yielding `idx` times string values
            for _ in range(idx):
                yield f"IDX: {idx}"

        def make_dataset(idx):
            # Create a tf.data.Dataset from the hello generator for each index
            return tf.data.Dataset.from_generator(
                lambda: hello(idx),
                output_signature=tf.TensorSpec(shape=(), dtype=tf.string),
            )

        # Create list of datasets ahead of time (to avoid autograph/graph-mode issues)
        datasets = [make_dataset(i) for i in range(N_DATASETS_TO_INTERLEAVE)]

        # Create a dataset from the dataset list and interleave them
        ds = tf.data.Dataset.from_tensor_slices(datasets)
        interleaved_ds = ds.interleave(lambda x: x, cycle_length=N_DATASETS_TO_INTERLEAVE)

        # To get output tensors from dataset, we must iterate it eagerly.
        # We'll gather results into a tf.TensorArray for returning.
        # Since datasets yield strings of shape (), dtype string, 
        # we'll gather and return a RaggedTensor of strings for all values.

        output_strings = []

        for elem in interleaved_ds:
            output_strings.append(elem)

        # Convert gathered strings list to a tf.RaggedTensor of shape (None,)
        return tf.ragged.constant(output_strings, dtype=tf.string)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # The model ignores input, but to respect signature we produce a dummy tensor input.
    # No particular shape since inputs are unused, providing scalar zero.
    return tf.constant(0)

