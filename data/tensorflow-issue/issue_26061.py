# tf.random.normal((None, 2), dtype=tf.float32) ← Input shape is (batch_size, 2) based on X_train shape

import tensorflow as tf

# Custom initializer implemented as a callable class for proper serialization in Keras
class MyGlorotInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = tf.float32
        stddev = tf.sqrt(2. / (shape[0] + shape[1]))
        return tf.random.normal(shape, stddev=stddev, dtype=dtype)
    
    def get_config(self):
        # Return config for serialization (empty since no params)
        return {}

# Custom regularizer implemented as a class to enable serialization
class MyL1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l1=0.01):
        self.l1 = l1
    
    def __call__(self, x):
        return tf.reduce_sum(tf.abs(self.l1 * x))
    
    def get_config(self):
        return {'l1': float(self.l1)}

# Custom constraint implemented as a class for proper serialization
class MyPositiveWeights(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.nn.relu(w)
    
    def get_config(self):
        return {}

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=MyGlorotInitializer(),
            kernel_regularizer=MyL1Regularizer(0.01),
            kernel_constraint=MyPositiveWeights()
        )
    
    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape (batch_size, 2)
    # Using batch_size = 4 as an example
    B = 4
    input_shape = (B, 2)
    return tf.random.normal(input_shape, dtype=tf.float32)

