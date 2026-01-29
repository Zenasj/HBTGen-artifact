# tf.random.uniform((2,), dtype=tf.int32) ← Inferred input shape from original minimal input example (shape=[2]), dtype inferred as int32

import tensorflow as tf
import attr

@attr.s
class WrappedTensor(object):
    tensor = attr.ib()

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights, purely identity/structural behavior based on the example

    def call(self, x):
        """
        Implements logic comparing behavior of distributing a dict vs an attr class wrapper.
        
        Given input x, which could be either 
          - a dict with key 'tensor' -> distributed Replicated input, or 
          - a WrappedTensor instance wrapping distributed tensors,
        this simulates the behavior relevant to the issue:
        
        When running tf.distribute.Strategy.run, dict input returns PerReplica inside the dict,
        but attr objects are wrapped differently: they are PerReplica of WrappedTensor unlike the dict case.
        
        This method returns a boolean Tensor indicating whether the treatment of PerReplica and WrappedTensor
        in the output differs from expected consistent nested structure behavior, i.e., whether wrapping order matches.
        
        For demonstration, we construct "expected" output like dict-case with PerReplica inside the container,
        and "actual" output like attr-case with container inside the PerReplica, then compare elementwise equality.
        """
        # Assuming input x is a dict {'tensor': ...} or WrappedTensor(tensor=...)
        # We create two outputs:
        # - dict_behavior: {'tensor': PerReplica(...)}
        # - attr_behavior: PerReplica(WrappedTensor(...))
        
        # Extract the tensor(s) to simulate their distributed versions
        # For simplicity, assume x is WrappedTensor or dict with key 'tensor'.
        
        # In actual TensorFlow distributed execution, PerReplica holds multiple replica tensors.
        # We mimic this here by splitting input along the batch dimension (batches=2).
        
        def simulate_per_replica(tensor):
            # Split into 2 replica tensors along batch axis 0 (assumed)
            replica_0 = tensor[0:1]
            replica_1 = tensor[1:2]
            return [replica_0, replica_1]
        
        if isinstance(x, dict):
            per_replicas = simulate_per_replica(x['tensor'])
            dict_behavior = {'tensor': per_replicas}  # dict wrapping list of replicas
            attr_behavior = [WrappedTensor(t) for t in per_replicas]  # list of WrappedTensor replicas
        elif isinstance(x, WrappedTensor):
            per_replicas = simulate_per_replica(x.tensor)
            dict_behavior = {'tensor': per_replicas}
            attr_behavior = [WrappedTensor(t) for t in per_replicas]
        else:
            # Fallback: treat as tensor alone
            per_replicas = simulate_per_replica(x)
            dict_behavior = {'tensor': per_replicas}
            attr_behavior = [WrappedTensor(t) for t in per_replicas]
        
        # Now compare:
        # In expected nested (dict) scenario, PerReplica is inside dict:
        # In actual attr scenario, attr object is wrapped inside PerReplica
        # We compare elements of dict_behavior['tensor'] with attr_behavior tensors
        
        # Equality means shapes and values should match (wrapped tensor values only)
        # Since WrappedTensor wraps tensor, compare tensors inside
        comparison_results = []
        for dtensor, atensor_wrapped in zip(dict_behavior['tensor'], attr_behavior):
            # Compare tensor equality elementwise, after squeezing batch axis
            # all close or equal check
            comparison_results.append(tf.reduce_all(tf.equal(dtensor, atensor_wrapped.tensor)))
        
        # Return a boolean tensor indicating if ALL elements match between dict_behavior and attr_behavior tensors
        # If True: they are equivalent, meaning wrapping difference is superficial
        # If False: they differ in structure or value (expected to be True here because values come from same slices)
        return tf.reduce_all(comparison_results)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tensor matching the original minimal input: tf.constant with shape [2], dtype int32
    # This corresponds to input accepted by MyModel.call (dict or WrappedTensor)
    # Here we provide a dict version to show consistent nested structure
    input_tensor = tf.random.uniform((2,), minval=0, maxval=10, dtype=tf.int32)
    # We return both wrapped forms to allow model input flexibility
    
    # For simpler interaction, return a dict with 'tensor' key, as that was the initial dict input in issue
    return {'tensor': input_tensor}

