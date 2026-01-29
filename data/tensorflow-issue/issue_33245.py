# tf.random.uniform((B, 32), dtype=tf.float32) ← Input shape inferred from example input_tensor shape (50, 32) in the issue.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_units, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.num_units = num_units
        # Two dense projection layers, as per the example
        self.dense_proj1 = tf.keras.layers.Dense(num_units, activation='relu')
        self.dense_proj2 = tf.keras.layers.Dense(num_units, activation='relu')

    def call(self, features):
        """
        Forward pass of the model.
        
        Args:
            features: dict containing key 'input' with a Tensor of shape (batch_size, 32).
        
        Returns:
            dict with keys 'proj1_output' and 'proj2_output' corresponding to outputs of 
            two separate dense layers applied to the input.
        """
        inp = features['input']
        proj1_output = self.dense_proj1(inp)
        proj2_output = self.dense_proj2(inp)
        return {
            'proj1_output': proj1_output,
            'proj2_output': proj2_output
        }

def my_model_function():
    # Instantiate the model with num_units=16 as per the example in the issue.
    # This matches the output dimension size of 16 in proj1_output and proj2_output.
    return MyModel(16)

def GetInput():
    """
    Returns a random tensor input dictionary expected by MyModel.
    The input shape is (batch_size=10, 32) of float32 dtype.

    This matches the batching indicated in the issue where dataset.batch(10) was used.
    """
    # Using batch size 10 as in the example dataset batching step.
    batch_size = 10
    input_shape = (batch_size, 32)
    random_input = tf.random.uniform(input_shape, dtype=tf.float32)
    return {'input': random_input}

