# tf.random.uniform((None, 3), dtype=tf.float32) ← Inferred input shape from example: vector with 3 elements, batch dimension None

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # No additional weights; model just indexes elements from input tensor

    def call(self, x):
        """
        Input x is expected to be shape (batch_size, 3)
        
        This replicates the example logic where output is a 2x2 tensor 
        constructed from specific elements of input x:
        
        OutputTensor[0,0] = x[0,0]
        OutputTensor[1,1] = x[1,0]
        OutputTensor[1,0] = x[2,0]
        
        Since batch dimension is dynamic, we gather elements across the batch.
        
        We handle batch dimension properly, assuming x shape is (batch_size, 3).
        To be consistent, we'll build an output tensor of shape (batch_size, 2, 2).
        
        If input batch size < 3, indexing x[2,0] etc will error,
        so this example assumes input batch dimension >= 3.
        """
        # Extract required elements from batch dimension:
        # x shape (batch_size, 3)
        # We pick first element of feature dimension for all indexing.
        # Gather values at batch indices 0,1,2 and feature 0
        val_00 = x[0, 0]  # scalar from batch idx 0, feature 0
        val_11 = x[1, 0]  # scalar from batch idx 1, feature 0
        val_10 = x[2, 0]  # scalar from batch idx 2, feature 0

        # Construct output tensor with shape (2,2)
        # Note: This example does not operate batch-wise but picks fixed batch slices
        output = tf.stack([
            [val_00, 0.0],
            [val_10, val_11]
        ])

        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel.
    # Input expected shape: (batch_size >= 3, 3)
    # We'll create batch size 3 for safe indexing.
    return tf.random.uniform((3, 3), dtype=tf.float32)

