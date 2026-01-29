# tf.random.uniform((B, 1), dtype=tf.int64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model replicates the structure from the issue:
    Input: shape=(1,), dtype=tf.int64
    Passes input through a StringLookup layer configured with vocabulary ['a', 'b']
    and then a Dense layer with 10 units.
    
    Note: The main reported issue was that the StringLookup layer loses its vocabulary upon model save/load.
    This model preserves the vocabulary explicitly as a class attribute to ensure loading works properly.
    """
    def __init__(self):
        super().__init__()
        # Store the vocabulary explicitly so it persists
        self.vocabulary = ['a', 'b']
        
        # StringLookup layer with fixed vocabulary, no trainable vocabulary setting
        # Using the experimental.preprocessing.StringLookup as in the original issue
        self.lookup = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.vocabulary,
            output_mode='int',
            mask_token=None
        )
        # Dense layer after lookup (assumed units=10 as in issue)
        self.dense = tf.keras.layers.Dense(10)
        
    def call(self, inputs):
        # inputs expected shape (B,1) int64 tensor of token IDs
        x = self.lookup(inputs)
        output = self.dense(x)
        return output

def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Returns a tensor input compatible with MyModel:
    - Shape (B, 1)
    - dtype int64
    - Values correspond to token IDs compatible with StringLookup layer which expects strings internally,
      but since input dtype=int64 was used in the issue, here we generate int64 indices representing tokens.
      
    Note: StringLookup is normally used on strings; however, the example used int64 inputs. 
    To match the code provided, we generate int64 indices of shape (batch_size, 1).
    
    For demonstration, we generate random int64 values within the vocabulary index range.
    Since vocabulary has 2 tokens plus 1 OOV token, indices are in [0, vocab_size),
    but StringLookup maps from string tokens; here input is int64 as in the issue.
    
    This is a best-effort assumption to keep input shape and dtype compatible.
    """
    batch_size = 4  # arbitrary batch size
    # Generate random integers between 0 and 1 (since vocabulary has 2 tokens)
    # to simulate token indices. Expand dims to get shape (B,1)
    return tf.random.uniform((batch_size, 1), minval=0, maxval=2, dtype=tf.int64)

