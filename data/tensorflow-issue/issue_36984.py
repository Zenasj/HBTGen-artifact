# tf.random.uniform((None,), dtype=tf.int32) ← Input shape inferred as a 1D tensor of integers matching the dataset elements (e.g., scalar integers)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the issue's insight to use tf.data.Options to disable autotune 
        # to avoid multiprocess hanging issues with tf.data.Dataset
        options = tf.data.Options()
        options.experimental_optimization.autotune = False
        
        # valid_handle corresponds to the "validation" dataset with options to mitigate multiprocessing issues
        self.valid_handle = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5]).with_options(options)
        
        # train_handle corresponds to the "training" dataset
        self.train_handle = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def call(self, inputs):
        # The original code uses multiprocessing.Process to consume datasets.
        # Since multiprocessing with TF dataset is complex and not safe inside tf.function,
        # here we simulate a simple forward pass:
        # Just check if inputs belong to train_handle and produce an output tensor indicating dataset membership
        
        # inputs are expected to be integers (batched or scalar).
        # For each input, output 1 if in train_handle else 0.
        train_elements = tf.constant([1,2,3,4,5,6,7,8,9], dtype=inputs.dtype)
        is_train = tf.reduce_any(tf.equal(tf.expand_dims(inputs, -1), train_elements), axis=-1)
        
        valid_elements = tf.constant([1,2,3,4,5], dtype=inputs.dtype)
        is_valid = tf.reduce_any(tf.equal(tf.expand_dims(inputs, -1), valid_elements), axis=-1)
        
        # Output dictionary with boolean masks for train and valid membership
        return {'is_train': is_train, 'is_valid': is_valid}


def my_model_function():
    # Return instance of MyModel
    return MyModel()


def GetInput():
    # Return random tensor input that works with MyModel
    # Since inputs are integers expected to be part of train_handle range,
    # generate a batch of 5 random integers in [1..9]
    return tf.random.uniform(shape=(5,), minval=1, maxval=10, dtype=tf.int32)

