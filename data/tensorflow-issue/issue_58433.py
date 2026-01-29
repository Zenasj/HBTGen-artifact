# tf.random.stateless_normal((B, ), seed=[int, int]) ← seed must be a pair of integers for stateless ops

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model demonstrates deterministic and per-replica reproducible random number 
    generation inside a tf.function using tf.random.stateless_normal.
    
    It simulates the common need to have a different but reproducible seed per replica 
    when using distributed strategies.
    
    The forward pass takes a scalar integer seed as input and simulates generating 
    a random normal vector per replica, using that seed combined with the replica id 
    to form the stateless RNG seed.
    
    Output is a dictionary mapping replica_id to its generated vector, represented 
    as a stacked tensor of shape (num_replicas, 3).
    
    Assumptions and notes:
    - Input is a tf.Tensor scalar integer seed.
    - Stateless RNG ops require a seed tensor of shape (2,) with integer dtype.
    - Demo uses MirroredStrategy if run in a distributed environment, else single replica.
    - This encapsulates the solution described in the issue for per-replica seeds inside tf.functions.
    """

    def __init__(self):
        super().__init__()
        # Use mirrored strategy if visible GPUs, else fallback to one replica
        try:
            self.strategy = tf.distribute.MirroredStrategy()
        except Exception:
            self.strategy = None

    def _per_replica_random(self, seed):
        """
        This function runs within a replica context; gets replica id and generates
        a stateless normal random tensor using seed + replica_id as seed.
        """
        replica_ctx = tf.distribute.get_replica_context()
        if replica_ctx is None:
            # No distribution context, single replica scenario
            # Use seed as is
            combined_seed = tf.stack([seed, seed])
        else:
            repl_id = replica_ctx.replica_id_in_sync_group
            combined_seed = tf.stack([seed + repl_id, seed + repl_id])
        # Fixed shape of [3] for demonstration as per issue examples
        sample = tf.random.stateless_normal(shape=(3,), seed=combined_seed)
        return sample

    @tf.function
    def call(self, seed):
        """
        Run distributed generation of random normals using stateless RNG.
        If strategy is available, run distributed, else single replica.

        Args:
          seed: scalar int32 tensor, base seed

        Returns:
          Tensor of shape (num_replicas, 3) with per-replica samples.
        """
        if self.strategy is not None:
            def replica_fn(seed):
                return self._per_replica_random(seed)
            per_replica_result = self.strategy.run(replica_fn, args=(seed,))
            # per_replica_result is a PerReplica object; gather values as a Tensor
            # This returns a tf.Tensor shaped (num_replicas, 3)
            gathered = self.strategy.experimental_local_results(per_replica_result)
            return tf.stack(gathered)
        else:
            return tf.expand_dims(self._per_replica_random(seed), axis=0)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random scalar seed integer tensor for MyModel input.
    # Using tf.random.uniform with shape () and int32 dtype for seed.
    # Range chosen arbitrary between 0 and 1000.
    return tf.random.uniform((), minval=0, maxval=1000, dtype=tf.int32)

