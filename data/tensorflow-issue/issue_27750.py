# tf.random.uniform((BATCHSIZE, SEQLEN), dtype=tf.float32) ← Inferred input shape from the RNN model example: input is a batch of sequences with length SEQLEN

import tensorflow as tf
from tensorflow.keras import layers as l

# Constants inferred from issue code snippets and context
SEQLEN = 16          # Sequence length (as per Reshape layer input)
RNN_CELLSIZE = 32    # Number of GRU units

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model layers matching the given Keras Sequential model
        
        # Reshape input shape [BATCHSIZE, SEQLEN] → [BATCHSIZE, SEQLEN, 1]
        self.reshape = l.Reshape([SEQLEN, 1], input_shape=[SEQLEN])
        
        # Two stacked GRU layers
        self.gru1 = l.GRU(RNN_CELLSIZE, return_sequences=True)
        self.gru2 = l.GRU(RNN_CELLSIZE)
        
        # Final Dense layer outputting single value
        self.dense = l.Dense(1)
        
        # Define Root Mean Square Error as metric (not needed for forward pass)
        # But kept here for completeness if compiling outside estimator
        def rmse(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
        self.rmse = rmse
    
    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.gru1(x, training=training)
        x = self.gru2(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching [batch_size, SEQLEN] dtype float32
    # Batch size chosen as 32 (common default), type float32 as GRU expects floats
    BATCHSIZE = 32
    return tf.random.uniform((BATCHSIZE, SEQLEN), dtype=tf.float32)

