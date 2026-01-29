# tf.random.uniform((B, T, 1), dtype=tf.float64)  # Based on input shape (batch, time_step=100, CityFactorNum=1)

import tensorflow as tf

class Linear(tf.keras.layers.Layer):
    def __init__(self, CityNum, CityFactorNum):   
        super(Linear, self).__init__()
        self.CityNum = CityNum    
        self.CityFactorNum = CityFactorNum

    def build(self, input_shape):
        # beta shape: (CityFactorNum, 1)
        self.beta  = self.add_weight(shape=(self.CityFactorNum, 1),  initializer='random_normal', trainable=True)
        # alpha shape: (CityNum,)
        self.alpha = self.add_weight(shape=(self.CityNum,),        initializer='random_normal', trainable=True)

    def call(self, X):
        # X shape assumed (batch, time, CityFactorNum)
        # Take X[:,1:] corresponding to time steps 1:end 
        # Add alpha broadcasted on batch dimension
        # Note: In original code, indexing uses tf.dtypes.cast(X[0,0], tf.int32), 
        # which seems like a bug or placeholder. We'll omit that cast part, as it is not meaningful for computation.
        # So implement the core linear form: tf.matmul(X[:,1:], beta) + alpha
        # Because X is time series of shape (batch, T, CityFactorNum), we consider X[:,1:,...]
        # We'll assume input X shape is (batch, time, CityFactorNum)
        # We apply matmul on last two dims: (batch, time-1, CityFactorNum) @ (CityFactorNum,1) => (batch, time-1, 1)
        x_slice = X[:, 1:, :]
        v = tf.matmul(x_slice, self.beta)  # shape (batch, time-1, 1)
        # alpha shape (CityNum,), broadcast on batch and time-1 dims as needed.
        # To add alpha with shape (CityNum,), we need to broadcast correctly.
        # But the input shape and alpha dimensions do not match time dimension. Original code is unclear here.
        # We assume CityNum corresponds to some feature dimension, so we need to broadcast or reduce dims.
        # Due to ambiguity, we simplify and just add alpha first element as scalar bias across all.
        # (This is a logical simplification since original behavior is not fully clear.)
        alpha_scalar = self.alpha[0] if self.CityNum > 0 else 0.0
        return v + alpha_scalar

class MinimalRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, CityNum, CityFactorNum, **kwargs):
        super(MinimalRNNCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = 1  # Fix to 1 matching new_state shape and get_initial_state
        self.CityNum = CityNum    
        self.CityFactorNum = CityFactorNum

    def build(self, input_shape):
        # input_shape is a list of shapes for the input tuple [X_input, U_input]
        # We infer shape for convenience if needed; here not strictly necessary.
        
        # Parameters beta and alpha shapes adapted to 1D vectors as per original
        self.beta  = self.add_weight(shape=(self.CityNum,),  initializer='random_normal', trainable=True)
        self.alpha = self.add_weight(shape=(self.CityFactorNum-1,), initializer='random_normal', trainable=True)
        
        # Several dense layers applied sequentially on U_input
        self.dense_1  = tf.keras.layers.Dense(32, activation='tanh')
        self.dense_2  = tf.keras.layers.Dense(64, activation='tanh')
        self.dense_3  = tf.keras.layers.Dense(64, activation='tanh')
        self.dense_4  = tf.keras.layers.Dense(64, activation='tanh')
        self.dense_5  = tf.keras.layers.Dense(1)
        
        self.Xbeta_Add_alpha = Linear(self.CityNum, self.CityFactorNum)
                
        super(MinimalRNNCell, self).build(input_shape)
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # Correct initial state shape: (batch_size, state_size)
        # state_size=1, so initial state shape = (batch_size, 1)
        # Use tf.ones with shape (batch_size, state_size)
        if batch_size is None:
            # fallback - dynamic batch size unknown, return tensor with shape (1, state_size)
            batch_size = 1
        if dtype is None:
            dtype = tf.float32
        initial_state = tf.ones((batch_size, self.state_size), dtype=dtype)
        return [initial_state]

    def call(self, inputs, states):
        # Inputs: tuple of two tensors: (X_input, U_input)
        # States: list with a single tensor of shape (batch_size, state_size=1)
        X_input = inputs[0]  # Shape (batch_size, time_step?, CityFactorNum)
        U_input = inputs[1]  # Shape (batch_size, time_step?, UFactorNum)
        
        s1 = states[0]  # shape (batch_size, 1)
        
        # Process U_input through 5 dense layers
        gU = self.dense_1(U_input)
        gU = self.dense_2(gU)
        gU = self.dense_3(gU)
        gU = self.dense_4(gU)
        gU = self.dense_5(gU)  # shape (batch_size, some_dim, 1)
        
        # Xbeta_Add_alpha expects input shape that matches linear layer
        X = self.Xbeta_Add_alpha(U_input)
        
        # Compute dot product along appropriate axes between gU and s1
        # - gU shape: (batch_size, some_dim, 1)
        # - s1 shape: (batch_size, 1)
        # The original code uses layers.dot with axes=1
        
        # We interpret this as batch-wise dot of last dims
        # Reshape s1 to match dims for dot: (batch_size, 1, 1)
        s1_expanded = tf.expand_dims(s1, axis=1)  # (batch_size, 1, 1)
        # Use tf.reduce_sum or tf.linalg.matmul to mimic dot
        # Multiply element-wise and sum on axis=1
        gUZ = tf.reduce_sum(gU * s1_expanded, axis=1, keepdims=True)  # shape (batch_size,1,1)

        gUZX = gUZ + tf.expand_dims(X, axis=1)  # broadcast add X to match shape
        
        output = [gUZ, gUZX]  # output is a list of two tensors
        new_state = [gUZX[:,0,:]]  # new_state shape (batch_size, 1)

        return output, new_state


class MyModel(tf.keras.Model):
    def __init__(self, CityNum=100, CityFactorNum=1, time_step=100, UFactorNum=1):
        super(MyModel, self).__init__()
        self.CityNum = CityNum
        self.CityFactorNum = CityFactorNum
        self.time_step = time_step
        self.UFactorNum = UFactorNum
        
        self.cell = MinimalRNNCell(1, self.CityNum, self.CityFactorNum)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)
        self.output_dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        # inputs is expected to be a list/tuple of two tensors: [X, U]
        # X shape: (batch, time_step, CityFactorNum)
        # U shape: (batch, time_step, UFactorNum)
        rnn_out = self.rnn(inputs)
        out = self.output_dense(rnn_out)
        return out

def my_model_function():
    # Return an instance of MyModel with default parameters matching example
    return MyModel()

def GetInput():
    # Creates valid random input tuple [X, U] matching MyModel input shapes
    batch_size = 1
    time_step = 100
    CityFactorNum = 1
    UFactorNum = 1

    # Use float64 as in original code
    X = tf.random.uniform((batch_size, time_step, CityFactorNum), dtype=tf.float64)
    U = tf.random.uniform((batch_size, time_step, UFactorNum), dtype=tf.float64)

    # Convert to float32 since layers Dense default to float32 unless overridden
    # Alternatively, make Dense layers use float64; 
    # For simplicity, cast inputs to float32 since Keras layers commonly expect that
    X = tf.cast(X, tf.float32)
    U = tf.cast(U, tf.float32)

    return [X, U]

