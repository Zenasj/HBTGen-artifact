# tf.random.uniform((50, 1), dtype=tf.float64)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Defensive check: units should be positive integer as per docs.
        # Here we set units=1 to avoid the bug observed when units=0.
        units = 1  
        activation = "sigmoid"
        use_bias = True
        kernel_initializer = "ones"
        bias_initializer = "zeros"
        kernel_regularizer = None
        bias_regularizer = None
        activity_regularizer = None
        kernel_constraint = None
        bias_constraint = None
        
        self.dense_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            dtype=tf.float64  # Match input dtype for consistency
        )
    
    def call(self, inputs):
        return self.dense_layer(inputs)

def my_model_function():
    # Return an instance of MyModel.
    # No pretrained weights needed here; the initializer sets up the weights.
    return MyModel()

def GetInput():
    # Generate a random input tensor compatible with the model.
    # Shape [50, 1] float64 as per original issue example.
    x = tf.random.uniform(
        shape=(50, 1), minval=1.0, maxval=3.0, dtype=tf.float64
    )
    return x

