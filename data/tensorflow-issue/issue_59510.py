# tf.random.uniform((3, 2, 2), dtype=tf.float32) ← inferred from the example batched_features shape (3, 2, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple example: pass input through a Dense layer for demonstration
        # Since original code only returned features directly, 
        # use a linear layer to simulate some transformation.
        self.dense = tf.keras.layers.Dense(4)  # arbitrary output dim
    
    def call(self, inputs, training=False):
        # According to the discussion and Keras semantics:
        # When using model.fit(dataset), the dataset yields (x, y) tuples.
        # model.fit automatically splits these into 'inputs' for .call and 'targets' for loss.
        # So in call(), inputs corresponds only to features, NOT (features, labels).
        # Therefore, attempting `feature, label = inputs` raises error.
        #
        # So here we just handle `inputs` as features tensor.
        #
        # For debugging, you could print inputs shape, but avoid iteration over tensor.
        # This satisfies XLA by avoiding unsupported operations.
        x = inputs

        # Apply some operation to simulate model logic.
        output = self.dense(x)
        return output


def my_model_function():
    # Return a ready-to-use MyModel instance.
    model = MyModel()
    # Compile with categorical crossentropy, assuming multi-class classification,
    # because labels seem categorical in original issue.
    # Also metrics match example.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    return model


def GetInput():
    # Produce a batch of inputs matching batched_features shape in the example
    # The original example used shape (3, 2, 2) but model.fit batches again.
    # So produce a single batch of shape (1, 2, 2) for direct usage with batch=1
    # Using float32 values to match typical input for Dense layer.
    # This dummy input matches the feature tensor expected by MyModel.
    # The label tensor should be numerical one-hot encoded corresponding to categorical labels.

    # Create features tensor: shape (1, 2, 2)
    features = tf.random.uniform((1, 2, 2), dtype=tf.float32)
    
    # Create labels as a batch of one-hot vectors corresponding to 2 classes (A, B)
    # Originally labels were string shape (3, 2, 1) with 'A' and 'B'.  
    # For dummy labels create shape (1, 2, num_classes=2)
    # Example: 'A' -> [1,0], 'B' -> [0,1]

    labels_raw = tf.constant([[0, 1]], dtype=tf.int32)  # dummy class indices for batch=1, seq_len=2
    labels = tf.one_hot(labels_raw, depth=2)  # shape (1, 2, 2)

    # Return tuple as expected by model.fit(dataset)
    return features, labels

