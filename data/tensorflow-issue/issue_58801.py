# tf.random.uniform((B, 100), dtype=tf.float32) ← Inferred input shape from Dense input_dim=100 in example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple Dense layer as per the issue example
        self.dense = tf.keras.layers.Dense(10)
        # Legacy Adam optimizer as recommended for backward compatibility in TF 2.11+
        self.optimizer_legacy = tf.keras.optimizers.legacy.Adam()
        # Experimental/default Adam optimizer from TF 2.11+, does not have `.weights` attribute
        self.optimizer_exp = tf.keras.optimizers.Adam()
    
    def call(self, inputs, training=False):
        # Forward pass through Dense layer
        outputs = self.dense(inputs)
        # Return outputs plus comparison info about optimizer weights presence
        
        # Check for `.weights` attribute existence in optimizers:
        legacy_has_weights = hasattr(self.optimizer_legacy, 'weights')
        exp_has_weights = hasattr(self.optimizer_exp, 'weights')
        
        # Compare weights attributes of legacy and experimental optimizers
        # This could represent diagnostic output for the issue context.
        optimizers_weights_match = tf.reduce_all(
            tf.equal(
                tf.constant(legacy_has_weights), 
                tf.constant(exp_has_weights)
            )
        )
        
        # Pack outputs and the boolean comparison into a dictionary for clarity
        return {
            "model_output": outputs,
            "legacy_has_weights": legacy_has_weights,
            "exp_has_weights": exp_has_weights,
            "optimizers_weights_match": optimizers_weights_match
        }

def my_model_function():
    # Return an instance of MyModel, using legacy Adam to avoid the issue
    # The model encapsulates both legacy and new optimizer behaviors to highlight the difference.
    return MyModel()

def GetInput():
    # Return random input tensor matching the expected input shape (batch dim arbitrary, e.g., 4)
    # Batch size chosen as 4 for example purposes.
    return tf.random.uniform((4, 100), dtype=tf.float32)

