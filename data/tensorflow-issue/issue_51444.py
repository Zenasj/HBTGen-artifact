# tf.ragged.constant([[B1 variable length], [B2 variable length], ...]) with ragged dims, batch variable, max inner length ~10

import tensorflow as tf
from tensorflow.keras.layers import Lambda

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # No trainable weights or layers in the original dummy model
        # Just casts the ragged input to float via a tensor intermediate,
        # then reconstructs ragged output matching the ragged input lengths.
        # The model demonstrates handling ragged tensor inputs and outputs.
        
    def call(self, inputs):
        # inputs: RaggedTensor with shape [batch, (variable length)]
        # Get lengths of each ragged row
        lens = tf.map_fn(lambda x: tf.shape(x)[0], inputs, fn_output_signature=tf.int32, name='get_lengths')
        # Convert ragged to padded dense tensor, pad value 0, max length = 10 (assumed from example)
        tensored_input = inputs.to_tensor(default_value=0, shape=[None, 10])
        # Cast to float32 (original used a Lambda layer for cast)
        x = tf.cast(tensored_input, dtype=tf.float32)
        # Convert back to ragged using original lengths to keep ragged structure
        outputs = tf.RaggedTensor.from_tensor(x, lengths=lens)
        return outputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Reproduce input similar to the example:
    # ragged batch of integer vectors of variable length max ~10
    ragged_data = tf.ragged.constant([
        [0,0,0,0,0],
        [0],
        [0,0,0],
        [0,0,0,0]
    ], ragged_rank=1)
    # This shape is (4, variable_length), batch size 4
    return ragged_data

