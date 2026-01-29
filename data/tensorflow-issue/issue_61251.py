# tf.random.uniform((B, 19, 19, 13), dtype=tf.float32) and tf.random.uniform((B, 7), dtype=tf.float32) are the input shapes inferred from saved_model_cli description

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No learnable parameters, the model returns a fixed set of 12 outputs for demonstration,
        # matching the example in the GitHub issue with outputs from 0 to 11.
        
    def call(self, inputs):
        # inputs is expected to be a tuple or list of two tensors matching
        # shapes: (batch_size, 19, 19, 13) and (batch_size, 7)
        # but we do not use inputs since example outputs are fixed scalars.
        
        batch_size = tf.shape(inputs[0])[0]
        
        # Create 12 outputs, each one a vector/tensor shaped per the example.
        # We mimic output shapes from the SavedModel signature shown:
        # outputs with various shapes:
        # output_1, output_2, output_9: (B, 362)
        # output_3, output_4: (B, 2)
        # output_5: (B, 19, 19, 1)
        # output_6, output_7: (B, 800)
        # output_8: (B, 1)
        # output_10, output_11, output_12: (B,)
        
        # To keep things simple, generate dummy tensors with values equal to output index
        # Broadcast scalar to the required shape for each output.
        
        output_1 = tf.fill([batch_size, 362], 1.0)
        output_2 = tf.fill([batch_size, 362], 2.0)
        output_3 = tf.fill([batch_size, 2], 3.0)
        output_4 = tf.fill([batch_size, 2], 4.0)
        output_5 = tf.fill([batch_size, 19, 19, 1], 5.0)
        output_6 = tf.fill([batch_size, 800], 6.0)
        output_7 = tf.fill([batch_size, 800], 7.0)
        output_8 = tf.fill([batch_size, 1], 8.0)
        output_9 = tf.fill([batch_size, 362], 9.0)
        output_10 = tf.fill([batch_size], 10.0)
        output_11 = tf.fill([batch_size], 11.0)
        output_12 = tf.fill([batch_size], 12.0)

        # Return outputs as a dictionary mapped by their numeric suffix keys,
        # to reflect realistic naming and to preserve order.
        # (The original issue states outputs come unordered by lexicographic sorting)
        return {
            "output_1": output_1,
            "output_2": output_2,
            "output_3": output_3,
            "output_4": output_4,
            "output_5": output_5,
            "output_6": output_6,
            "output_7": output_7,
            "output_8": output_8,
            "output_9": output_9,
            "output_10": output_10,
            "output_11": output_11,
            "output_12": output_12,
        }


def my_model_function():
    # Instantiate MyModel; weights are not needed since outputs are fixed
    return MyModel()


def GetInput():
    # Return a valid input tuple matching the input signature:
    # First input: (batch_size, 19, 19, 13), floats
    # Second input: (batch_size, 7), floats
    
    batch_size = 1  # Minimal batch size for testing, can be >1
    
    input_tensor_1 = tf.random.uniform(shape=(batch_size, 19, 19, 13), dtype=tf.float32)
    input_tensor_2 = tf.random.uniform(shape=(batch_size, 7), dtype=tf.float32)
    
    return (input_tensor_1, input_tensor_2)

