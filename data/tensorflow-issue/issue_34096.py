# tf.random.uniform((B, 2), dtype=tf.float32) ← Inferred input shape (batch_size, 2 features 'a' and 'b')

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define numeric feature columns for input features 'a' and 'b'
        self.columns = [
            tf.feature_column.numeric_column('a'),
            tf.feature_column.numeric_column('b')
        ]
        # DenseFeatures layer to process the input dictionary features
        self.input_column = keras.layers.DenseFeatures(self.columns)
        # Define simple feedforward layers as in the original example
        self.dense1 = keras.layers.Dense(4, activation='relu')
        self.dense2 = keras.layers.Dense(2)
    
    def call(self, inputs, training=False):
        """
        Expects inputs as a dictionary with keys 'a' and 'b', each a tensor of shape (batch_size,).
        Processes features via DenseFeatures layer and passes through dense layers.
        """
        # inputs is a dict { 'a': tensor, 'b': tensor }
        x = self.input_column(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    """
    Returns an instance of MyModel.
    """
    model = MyModel()
    # Compile the model with example loss and optimizer similar to original example
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
    return model

def GetInput():
    """
    Returns example input matching the expected input of MyModel.
    Since MyModel expects a dictionary of features 'a' and 'b', each shape (batch_size,),
    we provide batch_size=4 sample data as a dict of tensors.
    Tensor values are in float32 as this is the default dtype for features.
    """
    batch_size = 4
    # Random float inputs 0.0 to 1.0, matching numeric_column expectations
    a = tf.random.uniform((batch_size,), minval=0, maxval=1, dtype=tf.float32)
    b = tf.random.uniform((batch_size,), minval=0, maxval=1, dtype=tf.float32)
    
    # Return input as dictionary consistent with feature column input.
    return {'a': a, 'b': b}

