# tf.random.uniform((20, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST batch and time_steps, n_input

import tensorflow as tf

# Global constants inferred from original code and MNIST data shape
time_steps = 28  # MNIST image height (28 rows = time steps)
n_input = 28     # MNIST image width (28 columns = features per step)
n_hidden = 8     # hidden units for BasicLSTMCell
n_classes = 10   # number of output classes (digits 0-9)
batch_size = 20  # batch size used in example

# Trainable weights and biases as variables inside MyModel for clarity
# they correspond to output layer weights and biases from LSTM last output state
# Shape [n_hidden, n_classes], bias shape [n_classes]
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Create the BasicLSTMCell once in init, as advised in the comments
        self.lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Output layer weights and biases as tf.Variables
        # Using float32 dtype as per original
        self.out_weights = tf.Variable(
            initial_value=tf.random.normal([n_hidden, n_classes]), 
            trainable=True, dtype=tf.float32,
            name='out_weights')
        self.out_biases = tf.Variable(
            initial_value=tf.random.normal([n_classes]), 
            trainable=True, dtype=tf.float32,
            name='out_biases')
    
    def build(self, input_shape):
        # The LSTM cell expects input shape to build its variables
        # input_shape expected (batch_size, time_steps, n_input)
        self.lstm_cell.build(input_shape)
        super(MyModel, self).build(input_shape)  # Always call super build
    
    @tf.function
    def call(self, x):
        # Cast input to float32, as original code does
        x = tf.dtypes.cast(x, tf.float32, name='Converted_floats')
        
        # Unstack input into a list of tensors on axis 1 (time dimension)
        # This converts shape (B, T, C) -> list length T, each (B, C)
        x_unstacked = tf.unstack(x, num=time_steps, axis=1)
        
        # Run static_rnn with the cell and unstacked inputs
        # Returns all outputs and final state
        outputs, _ = tf.compat.v1.nn.static_rnn(cell=self.lstm_cell, inputs=x_unstacked, dtype=tf.float32)
        
        # Use the last output for classification
        last_output = outputs[-1]  # shape (batch_size, n_hidden)
        
        # Compute logits = last_output @ weights + bias
        logits = tf.matmul(last_output, self.out_weights) + self.out_biases
        
        return logits


def my_model_function():
    # Returns a new instance of MyModel with randomly initialized weights
    return MyModel()

def GetInput():
    # Return a random input tensor shaped (batch_size, time_steps, n_input)
    # dtype float32 matches model expectation
    return tf.random.uniform((batch_size, time_steps, n_input), dtype=tf.float32)

