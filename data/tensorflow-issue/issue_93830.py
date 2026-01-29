# tf.random.uniform((batch_size, 1), dtype=tf.string) ← Inputs are string tensors of shape (batch_size, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Vocabulary list based on example in the issue
        vocab_list = ["1", "2", "3", "cat", "dog", "mouse"]
        
        # String lookup layers convert string inputs to integer indices
        self.lookup_a = tf.keras.layers.StringLookup(
            vocabulary=vocab_list, mask_token=None, num_oov_indices=0, name='lookup_a')
        self.lookup_b = tf.keras.layers.StringLookup(
            vocabulary=vocab_list, mask_token=None, num_oov_indices=0, name='lookup_b')
        self.lookup_c = tf.keras.layers.StringLookup(
            vocabulary=vocab_list, mask_token=None, num_oov_indices=0, name='lookup_c')

        self.num_bins_crossing = 1000
        
        # HashedCrossing layers for crossing features
        self.cross_ab = tf.keras.layers.HashedCrossing(
            num_bins=self.num_bins_crossing, name='cross_ab')
        self.cross_abc = tf.keras.layers.HashedCrossing(
            num_bins=self.num_bins_crossing, name='cross_abc')

        # Embedding and output layers
        self.embedding_dim = 8
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.num_bins_crossing, output_dim=self.embedding_dim, name='embedding')
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')

    def call(self, inputs):
        # inputs: list or tuple of three tensors: [feature_a, feature_b, feature_c]
        feature_a, feature_b, feature_c = inputs

        # Apply string lookup (output shape: (batch_size, 1), dtype=int64)
        idx_a = self.lookup_a(feature_a)
        idx_b = self.lookup_b(feature_b)
        idx_c = self.lookup_c(feature_c)

        # HashedCrossing requires all inputs to have same shape, 
        # but the issue is that the first cross output shape is (None, 1) symbolic
        # while idx_c often resolves to (batch_size,1) concretely during fit.
        # To avoid shape mismatch, explicitly set shape of cross_ab:
        cross_ab_out = self.cross_ab([idx_a, idx_b])
        cross_ab_out = tf.ensure_shape(cross_ab_out, [None, 1])  # ensure symbolic shape with batch dim None

        # Second cross: cross (A x B) with C
        cross_abc_out = self.cross_abc([cross_ab_out, idx_c])

        # Embedding and prediction
        x = self.embedding(cross_abc_out)
        x = self.flatten(x)
        output = self.output_layer(x)

        return output


def my_model_function():
    # Create inputs mimicking the Functional API inputs from the issue for clarity
    input_a = tf.keras.Input(shape=(1,), dtype=tf.string, name='feature_a')
    input_b = tf.keras.Input(shape=(1,), dtype=tf.string, name='feature_b')
    input_c = tf.keras.Input(shape=(1,), dtype=tf.string, name='feature_c')

    # Instantiate MyModel and call on inputs
    model_instance = MyModel()
    outputs = model_instance([input_a, input_b, input_c])

    # Create functional model to keep API consistent if needed downstream
    # but we return the subclass instance as required by the task
    # (Functional model here is for completeness; return subclass instance)
    return model_instance


def GetInput():
    import numpy as np

    # Vocabulary list consistent with model initialization
    vocab_list = ["1", "2", "3", "cat", "dog", "mouse"]
    
    batch_size = 10  # typical batch size used in the issue example
    shape = (batch_size, 1)

    # Generate random string inputs consistent with vocab and shape
    dummy_a_np = np.random.choice(vocab_list, size=shape).astype(object)
    dummy_b_np = np.random.choice(vocab_list, size=shape).astype(object)
    dummy_c_np = np.random.choice(vocab_list, size=shape).astype(object)

    # Return tuple of tf.Tensor inputs of dtype string
    inp_a = tf.convert_to_tensor(dummy_a_np, dtype=tf.string)
    inp_b = tf.convert_to_tensor(dummy_b_np, dtype=tf.string)
    inp_c = tf.convert_to_tensor(dummy_c_np, dtype=tf.string)

    return (inp_a, inp_b, inp_c)

