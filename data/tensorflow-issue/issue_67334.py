# tf.random.uniform((B, ..., ...), dtype=tf.float32) ← Input shape is dynamic / unknown, inferred from extension type usage

import tensorflow as tf

# Reconstructing the ExtensionType Network as a submodule
class Network(tf.experimental.BatchableExtensionType):
    shape: tf.TensorShape  # batch shape. A single network has shape=[]
    work: tf.Tensor        # work[*shape, n] = work left to do at node n
    bandwidth: tf.Tensor   # bandwidth[*shape, n1, n2] = bandwidth from n1->n2

    def __init__(self, work, bandwidth):
        self.work = tf.convert_to_tensor(work)
        self.bandwidth = tf.convert_to_tensor(bandwidth)
        # Merge batch shapes from work and bandwidth tensors (except last dims)
        work_batch_shape = self.work.shape[:-1]
        bandwidth_batch_shape = self.bandwidth.shape[:-2]
        self.shape = work_batch_shape.merge_with(bandwidth_batch_shape)

    def __repr__(self):
        # Simple string repr showing shapes
        return f"Network(work shape={self.work.shape}, bandwidth shape={self.bandwidth.shape})"


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # In the original example, the model is just a Sequential with an Input layer for Network.Spec
        # Since tf.keras.layers.Input lost support for type_spec argument in TF 2.16,
        # we simulate the Network embedding as part of this model's forward logic.
        # You could extend this model with custom layers processing Network objects.
        
        # For demonstration, no extra layers (placeholder identity)
        self.identity = tf.keras.layers.Lambda(lambda x: x)

    def call(self, network: Network, training=False):
        # Forward pass expects a Network ExtensionType instance
        # Since Keras does not directly support type_spec inputs anymore,
        # assume input is constructed externally as a valid Network object.
        # Simply pass through the input's work and bandwidth tensors concatenated for demonstration
        
        # Example logic: concatenate work and bandwidth along last dimension as a dummy operation
        # Note: batch dims may exist, so flatten to 2D for concat if needed
        work_shape = tf.shape(network.work)
        bandwidth_shape = tf.shape(network.bandwidth)
        
        # Flatten batch dims if any
        work_flat = tf.reshape(network.work, [work_shape[0], -1]) if tf.rank(network.work) > 2 else network.work
        bandwidth_flat = tf.reshape(network.bandwidth, [bandwidth_shape[0], -1]) if tf.rank(network.bandwidth) > 3 else network.bandwidth
        
        # Concatenate work and bandwidth flat tensors along last axis
        output = tf.concat([work_flat, bandwidth_flat], axis=-1)
        
        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Construct a random valid input matching the Network ExtensionType expected by MyModel
    # Since shape is dynamic, assume batch size 2, n=3 nodes for work and bandwidth dimension sizes
    
    B = 2  # batch size (arbitrary)
    n = 3  # number of nodes
    # work shape: [B, n]
    work = tf.random.uniform(shape=(B, n), dtype=tf.float32)
    # bandwidth shape: [B, n, n]
    bandwidth = tf.random.uniform(shape=(B, n, n), dtype=tf.float32)
    
    # Create the Network instance
    input_network = Network(work=work, bandwidth=bandwidth)
    return input_network

