# tf.random.uniform((512, 20, 15), dtype=tf.float32) ← inferred batch size = 512, frames = 20, joints*dim = 15 (5*3)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # output units: target frames * joints * dim = 2 * 5 * 3 = 30
        self.dense = tf.keras.layers.Dense(30)
        # reshape output to (f2=2, J=5, dim=3)
        self.reshape = tf.keras.layers.Reshape((2, 5, 3))
        
        # Define constant limbs for loss calculations: list of index pairs
        # The limb pairs represent connected joints on the skeleton
        self.limbs = tf.constant([
            (0, 1), (1, 2), (2, 3), (3, 4)
        ], dtype=tf.int32)

    def call(self, inputs, training=False):
        """
        Forward pass of the model: input shape (bs, f1, J*dim).
        Output shape: (bs, f2, J, dim), as per the original code.
        """
        x = self.flatten(inputs)
        x = self.dense(x)
        x = self.reshape(x)
        return x

    def mean_limb_length(self, y_true, y_pred):
        """
        Custom loss function that calculates the mean absolute difference
        of limb lengths between predicted and true poses.
        Expects shapes: (batchsize, frames, J * 3)
        """
        def get_limb_lengths(person, limbs):
            # person shape: (J*3,) or (J,3)
            person = tf.reshape(person, (-1, 3))  # reshape to (J, 3)
            # For each limb (joint pair), compute euclidean distance
            distances = tf.map_fn(
                lambda limb: tf.sqrt(
                    tf.reduce_sum(tf.square(
                        person[limb[0]] - person[limb[1]]
                    ))
                ),
                limbs,
                dtype=tf.float32
            )
            return distances

        def mean_limb_length_per_sequence(y_t, y_p, limbs):
            # y_t, y_p shape: (frames, J*3)
            # Compute mean absolute difference of limb lengths over frames
            diff = tf.map_fn(
                lambda x: tf.reduce_mean(
                    tf.abs(get_limb_lengths(x[0], limbs) - get_limb_lengths(x[1], limbs))
                ),
                (y_t, y_p),
                dtype=tf.float32
            )
            return diff

        def mean_limb_length_on_batch(y_t, y_p, limbs):
            # y_t, y_p shape: (batchsize, frames, J*3)
            # Compute mean limb length difference per batch element
            loss = tf.map_fn(
                lambda x: tf.reduce_mean(
                    mean_limb_length_per_sequence(x[0], x[1], limbs)
                ),
                (y_t, y_p),
                dtype=tf.float32
            )
            return tf.reduce_mean(loss)

        return mean_limb_length_on_batch(y_true, y_pred, self.limbs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Produce a random input tensor matching (batchsize=512, f1=20, J*dim=15)
    # matching the shape used in the original example (512, 20, 5*3)
    bs = 512
    f1 = 20
    J = 5
    dim = 3
    return tf.random.uniform(shape=(bs, f1, J * dim), dtype=tf.float32)

