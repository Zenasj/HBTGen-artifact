# tf.random.uniform((B, p), dtype=tf.float32) ← assuming input shape (batch_size, p) as per original code

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Combined model that encapsulates both base model (f0) and staged model (f1).
    The base model maps input vector of shape (p,) through Dense layers to a scalar output.
    The staged model takes two inputs:
      - output (penultimate layer) of base model (f0)
      - original input x
    and concatenates them, then continues with Dense layers and outputs a scalar.

    The forward call returns a tuple of (f0_output, f1_output), which matches the original separate models.

    This combined approach reflects the relationships in the issue where f1 is trained on [x, x]
    and based on frozen f0 outputs.

    Assumptions / notes:
    - Using tf.keras.layers.ELU activation as 'elu' in original.
    - Using SGD optimizer for both as per user requirements.
    - Input feature dimension 'p' is a parameter.
    - For frozen base model layers in f1, set trainable=False.
    """

    def __init__(self, p, units=20):
        super().__init__()
        self.p = p
        self.units = units
        
        # Base model components (f0)
        # Input: (batch, p)
        # Architecture from create_base_model:
        # Input -> Dense(units, 'elu') -> Dense(p) -> Dense(1) for training output
        # We separate penultimate_layer output to reuse in f1.
        self.base_input = tf.keras.layers.InputLayer(input_shape=(p,))
        self.base_dense1 = tf.keras.layers.Dense(units, activation='elu')
        self.base_dense2 = tf.keras.layers.Dense(p)
        self.base_output_layer = tf.keras.layers.Dense(1)
        
        # Build base model layers as components (for forward reuse)
        
        # Staged model components (f1)
        # Inputs: frozen base model output (penultimate_layer) and original input (p,)
        # Architecture from create_staged_model:
        # Concatenate [base_model_output, input] -> Dense(units, 'elu') -> Dense(p) -> Dense(1)
        # Freeze base model layers to simulate freezing of layers.
        
        self.freeze_base_layers = False  # Will be set True during staged forward
        
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.staged_dense1 = tf.keras.layers.Dense(units, activation='elu')
        self.staged_dense2 = tf.keras.layers.Dense(p)
        self.staged_output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        """
        Forward pass supports two modes based on input:
        1) Single input tensor of shape (batch, p) for base model (f0) only.
           Returns single output tensor (batch, 1).

        2) Tuple/list of two tensors: (x_for_f0, x_for_f1), both shape (batch, p).
           - f0 output penultimate layer (before output layer) is computed.
           - f1 uses frozen f0 penultimate output + second input,
             concatenates and continues staged model forward.
           Returns tuple (f0_output, f1_output).
        """
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            # Staged model forward: inputs = [x_for_f0, x_for_f1]
            x0, x1 = inputs
            
            # Base model forward to penultimate layer (frozen)
            z0 = self.base_input(x0)
            z0 = self.base_dense1(z0)
            z0 = self.base_dense2(z0)
            
            # Freeze base model layers effect simulated via no trainable weight updates from here
            # (But we do not change trainability flags here since it's dynamic call)

            # f0 output layer for reference output (optional)
            out0 = self.base_output_layer(z0)
            
            # Staged model forward
            z_concat = self.concat_layer([z0, x1])
            z = self.staged_dense1(z_concat)
            z = self.staged_dense2(z)
            out1 = self.staged_output_layer(z)
            
            return out0, out1

        else:
            # Base model forward only (inputs is tensor (batch, p))
            x = inputs
            z = self.base_input(x)
            z = self.base_dense1(z)
            z = self.base_dense2(z)
            out = self.base_output_layer(z)
            return out

def my_model_function():
    """
    Returns an instance of MyModel.
    Assumes p (input feature dimension) is 100 as a reasonable default for demonstration.
    """
    p = 100  # Assumed feature dimension
    model = MyModel(p, units=20)
    
    # Compile with SGD optimizer and MSE loss to match original setup.
    # This is required since original compiled models l use optimizer and loss.
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mse')
    return model

def GetInput():
    """
    Returns a random input that matches expected input for MyModel.
    Since MyModel can accept either a single input tensor or a pair of tensors,
    we return a tuple of two inputs for staged model:
        - x0: input tensor of shape (batch_size, p)
        - x1: input tensor of shape (batch_size, p)

    Using batch_size=32 and p=100 as assumed input shape.
    """
    batch_size = 32
    p = 100
    x0 = tf.random.uniform((batch_size, p), dtype=tf.float32)
    x1 = tf.random.uniform((batch_size, p), dtype=tf.float32)
    return (x0, x1)

