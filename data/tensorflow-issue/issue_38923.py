# tf.random.uniform((B, 32), dtype=tf.float32) ← Input shape inferred as (batch_size, 32) based on code snippets with Input(shape=(32,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Simple Dense layer similar to provided example
        self.dense = tf.keras.layers.Dense(32)

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        return x

    def save(self, filepath, save_format='tf', **kwargs):
        # Use standard save method - provide wrapper for consistency with issue context
        super().save(filepath=filepath, save_format=save_format, **kwargs)

    def load(self, filepath, load_format='tf'):
        """Load weights or model consistent with save/load_format.

        This is a helper method inspired by the discussion in the issue:
        - If loading a TensorFlow SavedModel directory (TF format), uses load_weights()
          with path to variables/variables checkpoint as workaround.
        - If loading HDF5 weights (h5 format), uses load_weights directly.
        - This allows usage of the subclassed MyModel instance to reload weights
          maintaining the custom class type, unlike standard keras.models.load_model.
        """
        if load_format == 'tf':
            # TensorFlow checkpoint format loading requires the full path to checkpoint files inside SavedModel folder
            # This is a workaround based on issue: load_weights('./saved_model/variables/variables')
            # Assumes that `filepath` is the SavedModel directory
            checkpoint_path = filepath + "/variables/variables"
            self.load_weights(checkpoint_path)
        elif load_format == 'h5':
            # Load HDF5 weights file directly
            self.load_weights(filepath)
        else:
            raise ValueError(f"Unsupported load_format '{load_format}'. Use 'tf' or 'h5'.")

def my_model_function():
    # Return an instance of MyModel with the appropriate Dense layer initialized
    model = MyModel()
    # Build the model by calling once with an input shape
    # This is needed for subclassed models to create weights before saving/loading
    dummy_input = tf.zeros((1, 32))
    model(dummy_input)
    return model

def GetInput():
    # Return a random input tensor matching expected input to MyModel: shape (batch_size, 32)
    # Assume batch_size=4 for example
    return tf.random.uniform((4, 32), dtype=tf.float32)

